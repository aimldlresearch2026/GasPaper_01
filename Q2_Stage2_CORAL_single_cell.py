# Q2 Stage 2 — Single-cell Kaggle script: CORAL + quantum kernel SVM
# Dataset: UCI Gas Sensor Array Drift; transductive, unlabeled-target adaptation
import os, sys, subprocess, zipfile, io, re, json, time, random, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
for module, package in [('pennylane','pennylane'),('requests','requests')]:
    try: __import__(module)
    except ImportError: subprocess.check_call([sys.executable,'-m','pip','install','-q',package])
import numpy as np, pandas as pd, requests, torch, pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# -------- CONFIG --------
SEEDS=[42,123,2026]       # change to [42,123,2026,7,99] for final study
QUBITS=4                 # 4 recommended; 6 increases quantum simulation cost
Q_DEPTH=2
N_TRAIN=400              # QSVM kernel is quadratic in training set size
N_VAL=200
N_TEST=240
RIDGE=1e-3
OUT=Path('/kaggle/working/Q2_Stage2_CORAL_Research')
OUT.mkdir(parents=True,exist_ok=True)
print('PennyLane',qml.__version__,'| Torch CUDA',torch.cuda.is_available(),'| quantum state simulator: CPU')

# -------- LOAD UCI DATA --------
def nested_dat(data, depth=0):
    if depth>5: return []
    out=[]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for item in z.infolist():
            if item.is_dir(): continue
            if item.filename.lower().endswith('.dat'): out.append((item.filename,z.read(item)))
            elif item.filename.lower().endswith('.zip'):
                try: out.extend(nested_dat(z.read(item),depth+1))
                except Exception: pass
    return out
files=[]
for p in Path('/kaggle/input').rglob('*'):
    if not p.is_file(): continue
    if p.suffix.lower()=='.dat': files.append((str(p),p.read_bytes()))
    elif p.suffix.lower()=='.zip':
        try: files.extend(nested_dat(p.read_bytes()))
        except Exception: pass
if not files:
    for url in ['https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip',
                'https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip']:
        try:
            r=requests.get(url,timeout=90);r.raise_for_status();files=nested_dat(r.content)
            if files: break
        except Exception as e: print('Download unavailable:',str(e)[:120])
if not files: raise RuntimeError('Attach the official UCI Gas Sensor Array Drift ZIP to Kaggle, or enable Internet.')
# Avoid duplicate data if multiple copies were attached
unique={}
for name,blob in files:
    match=re.search(r'batch[_\s-]*(\d+)',name,re.I)
    if match: unique[int(match.group(1))]=(name,blob)
if len(unique)<10: raise RuntimeError(f'Expected 10 batch files, found {sorted(unique)}. Attach complete UCI dataset.')
X=[]; Y=[]; B=[]
for b,(name,blob) in sorted(unique.items()):
    for line in blob.decode('utf-8',errors='replace').splitlines():
        tokens=line.split()
        if not tokens: continue
        try:
            lab=int(float(tokens[0]));vec=np.zeros(128,dtype=np.float64)
            for tok in tokens[1:]:
                if ':' in tok:
                    j,v=tok.split(':',1);j=int(j)-1
                    if 0<=j<128: vec[j]=float(v)
            X.append(vec);Y.append(lab);B.append(b)
        except (ValueError,IndexError): continue
X=np.asarray(X); Y=np.asarray(Y); B=np.asarray(B)
classes=np.unique(Y); mapping={c:i for i,c in enumerate(classes)}; y=np.array([mapping[v] for v in Y])
assert len(classes)==6, f'Expected 6 classes, got {classes}'
print('Dataset:',X.shape,'batch sizes:',dict(zip(*np.unique(B,return_counts=True))))

# -------- SPLITS AND CORAL --------
source=np.where(B<=7)[0]; val=np.where(B==8)[0]; test=np.where(B>=9)[0]
assert len(source)>0 and len(val)>0 and len(test)>0

def sample_ids(ids,n,seed):
    if len(ids)<=n: return ids
    try: picked,_=train_test_split(ids,train_size=n,random_state=seed,stratify=y[ids]);return np.sort(picked)
    except ValueError: return np.sort(np.random.default_rng(seed).choice(ids,n,replace=False))

def matrix_power_sym(A,power,eps=1e-9):
    w,v=np.linalg.eigh((A+A.T)/2)
    return (v*np.maximum(w,eps)**power)@v.T

def coral_source_to_target(src,tgt,ridge=RIDGE):
    # Align labeled source features to the covariance of UNLABELED target features.
    # No target labels are used. The target feature set is transductively available.
    sm=src.mean(axis=0);tm=tgt.mean(axis=0)
    cs=np.cov(src,rowvar=False)+ridge*np.eye(src.shape[1])
    ct=np.cov(tgt,rowvar=False)+ridge*np.eye(src.shape[1])
    return (src-sm)@matrix_power_sym(cs,-0.5)@matrix_power_sym(ct,0.5)+tm

# quantum feature map, statevector overlaps (pure-state fidelity)
qdev=qml.device('default.qubit',wires=QUBITS)
@qml.qnode(qdev)
def q_state(x):
    for layer in range(Q_DEPTH):
        for wire in range(QUBITS):
            qml.RY(x[wire]*(layer+1),wires=wire)
            qml.RZ(0.5*x[wire],wires=wire)
        for wire in range(QUBITS-1): qml.CNOT(wires=[wire,wire+1])
    return qml.state()

def embed(arr):
    return np.asarray([q_state(row) for row in arr],dtype=np.complex128)

def quantum_gram(A,B):
    return np.abs(A@B.conj().T)**2

# All four models use the SAME train, validation, and test subsets.
# Model selection on batch 8; final test batches 9-10 evaluated only afterward.
records=[]; predictions=[]; chosen=[]
for seed in SEEDS:
    print('\n'+'='*65+'\nSeed',seed)
    tr=sample_ids(source,N_TRAIN,seed)
    va=sample_ids(val,N_VAL,seed)
    te=sample_ids(test,N_TEST,seed)
    # Source-only feature fitting prevents target distribution leakage in preprocessing.
    scaler=StandardScaler().fit(X[tr]); xs=scaler.transform(X[tr]);xv=scaler.transform(X[va]);xt=scaler.transform(X[te])
    pca=PCA(n_components=QUBITS,random_state=seed).fit(xs)
    xs=pca.transform(xs);xv=pca.transform(xv);xt=pca.transform(xt)
    # Normalize using SOURCE training statistics only, then clip quantum angles.
    center=xs.mean(0);spread=xs.std(0)+1e-6
    xs=np.clip((xs-center)/spread,-3,3)
    xv=np.clip((xv-center)/spread,-3,3)
    xt=np.clip((xt-center)/spread,-3,3)
    # Unlabeled target feature adaptation, fitted separately for validation and test.
    # Using test FEATURES is transductive; no test LABELS are accessed for CORAL.
    xs_val_c=coral_source_to_target(xs,xv)
    xs_test_c=coral_source_to_target(xs,xt)
    features={
        'SVM':(xs,xv,xs,xt),
        'SVM_CORAL':(xs_val_c,xv,xs_test_c,xt),
        'QSVM':(xs,xv,xs,xt),
        'QSVM_CORAL':(xs_val_c,xv,xs_test_c,xt),
    }
    for name,(trv,valv,trt,tst) in features.items():
        quantum=name.startswith('QSVM')
        best=None
        # Scale features to angles for quantum models. Classical SVM uses same angles
        # so that kernel family, rather than input scale, is the key difference.
        trv=trv*np.pi/3;valv=valv*np.pi/3;trt=trt*np.pi/3;tst=tst*np.pi/3
        t0=time.time()
        if quantum:
            # Precompute quantum statevectors once per split/configuration.
            sv=embed(trv);vv=embed(valv)
            Ktr=quantum_gram(sv,sv);Kval=quantum_gram(vv,sv)
        for C in [0.1,1,10]:
            if quantum:
                model=SVC(C=C,kernel='precomputed',class_weight='balanced')
                model.fit(Ktr,y[tr]); pv=model.predict(Kval)
            else:
                model=SVC(C=C,kernel='rbf',gamma='scale',class_weight='balanced')
                model.fit(trv,y[tr]);pv=model.predict(valv)
            score=f1_score(y[va],pv,average='macro',zero_division=0)
            if best is None or score>best[0]: best=(score,C)
        val_f1,C=best
        if quantum:
            st=embed(trt);tt=embed(tst)
            Ktrain=quantum_gram(st,st);Ktest=quantum_gram(tt,st)
            clf=SVC(C=C,kernel='precomputed',class_weight='balanced').fit(Ktrain,y[tr]);pred=clf.predict(Ktest)
        else:
            clf=SVC(C=C,kernel='rbf',gamma='scale',class_weight='balanced').fit(trt,y[tr]);pred=clf.predict(tst)
        duration=time.time()-t0
        result={'seed':seed,'model':name,'C':C,'validation_macro_f1':val_f1,
                'accuracy':accuracy_score(y[te],pred),'macro_f1':f1_score(y[te],pred,average='macro',zero_division=0),
                'macro_precision':precision_score(y[te],pred,average='macro',zero_division=0),
                'macro_recall':recall_score(y[te],pred,average='macro',zero_division=0),
                'seconds':duration,'train_n':len(tr),'val_n':len(va),'test_n':len(te)}
        records.append(result)
        print(name,'C',C,'val F1',round(val_f1,4),'test acc',round(result['accuracy'],4),'test F1',round(result['macro_f1'],4),'sec',round(duration,1))
        for idx,p in zip(te,pred): predictions.append({'seed':seed,'model':name,'row_index':int(idx),'batch':int(B[idx]),'true':int(y[idx]),'pred':int(p)})
        if seed==SEEDS[0]:
            cm=confusion_matrix(y[te],pred,labels=np.arange(6))
            fig,ax=plt.subplots(figsize=(7,6))
            ConfusionMatrixDisplay(cm,display_labels=[str(c) for c in classes]).plot(ax=ax,cmap='Blues',colorbar=False)
            ax.set_title(f'{name} | Chronological drift | seed {seed}');plt.tight_layout()
            fig.savefig(OUT/f'confusion_{name}.png',dpi=180);plt.close(fig)
            (OUT/f'report_{name}.txt').write_text(classification_report(y[te],pred,labels=np.arange(6),zero_division=0))
    pd.DataFrame(records).to_csv(OUT/'all_seed_results.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'predictions.csv',index=False)

res=pd.DataFrame(records)
summary=res.groupby('model').agg(accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),runtime_mean=('seconds','mean')).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
# Paired per-seed improvements, NOT independent-sample significance tests.
wide=res.pivot(index='seed',columns='model',values='macro_f1')
paired=pd.DataFrame({'seed':wide.index,'CORAL_minus_SVM':wide['SVM_CORAL']-wide['SVM'],
                     'CORAL_minus_QSVM':wide['QSVM_CORAL']-wide['QSVM'],
                     'QSVM_CORAL_minus_SVM_CORAL':wide['QSVM_CORAL']-wide['SVM_CORAL']})
paired.to_csv(OUT/'paired_differences.csv',index=False)
fig,ax=plt.subplots(figsize=(9,5));ax.bar(summary['model'],summary['macro_f1_mean'],yerr=summary['macro_f1_sd'].fillna(0),capsize=4)
ax.set_ylim(0,1);ax.set_ylabel('Mean macro F1 (± SD across seeds)');ax.set_title('Q2 Stage 2: Chronological drift comparison');plt.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
metadata={'dataset':'UCI Gas Sensor Array Drift','splits':{'source':'batches 1-7','validation':'batch 8','test':'batches 9-10'},
          'adaptation':'transductive CORAL, target features only, no target labels',
          'seeds':SEEDS,'qubits':QUBITS,'depth':Q_DEPTH,'train_subset':N_TRAIN,'validation_subset':N_VAL,'test_subset':N_TEST,
          'kernel':'squared quantum state overlap; noiseless statevector simulator',
          'limitation':'This evaluates kernel-based CORAL adaptation, not hardware advantage; test features used unlabeled (transductive).'}
(OUT/'experiment_config.json').write_text(json.dumps(metadata,indent=2))
zip_path=Path('/kaggle/working/Q2_Stage2_CORAL_Research.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file(): z.write(p,arcname=p.relative_to(OUT))
print('\nFINAL SUMMARY\n',summary.to_string(index=False))
print('\nSaved:',zip_path)
try:
    from IPython.display import display,FileLink
    display(FileLink(str(zip_path)))
except Exception: pass
