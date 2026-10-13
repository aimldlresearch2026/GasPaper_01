# Q2 STAGE 3 — ONE CELL KAGGLE RESEARCH EXPERIMENT
# Chronological gas sensor drift: SVM vs quantum-kernel SVM; None / CORAL / MMD
# Unlabeled target features used for transductive adaptation. NEVER target labels.
import os, io, re, json, time, zipfile, hashlib, warnings, random
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.stats import t as student_t, wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics.pairwise import rbf_kernel

# ==================== USER CONFIGURATION ====================
SEEDS = [42, 123, 2026, 7, 99]
QUBITS = 4             # Phase A: 4; run 6 or 8 separately for ablations
Q_DEPTH = 2            # Phase A: 2
N_TRAIN = 400          # Matched training sample count for both classifiers
N_VAL = 200            # Batch 8: validation labels only, never final-test labels
N_TEST = None          # None = ALL samples in batches 9 and 10
C_GRID = [0.1, 1.0, 10.0]
MMD_STEPS = 120        # Affine feature alignment, unsupervised
MMD_LR = 0.025
MMD_REG = 0.12
MMD_FIT_TARGET_MAX = 900   # Unlabeled target subsample for adaptation only
RIDGE = 1e-3
OUTPUT = Path('/kaggle/working/Q2_Stage3_Research') if Path('/kaggle').exists() else Path('Q2_Stage3_Research')
OUTPUT.mkdir(parents=True, exist_ok=True)
print('Q2 Stage 3 | output:', OUTPUT.resolve())

# ==================== DATA LOADER ====================
def nested_dat(blob, depth=0):
    if depth > 5: return []
    found=[]
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for item in z.infolist():
            if item.is_dir(): continue
            if item.filename.lower().endswith('.dat'):
                found.append((item.filename,z.read(item)))
            elif item.filename.lower().endswith('.zip'):
                try: found.extend(nested_dat(z.read(item),depth+1))
                except (zipfile.BadZipFile, ValueError): pass
    return found

files=[]
for root in ['/kaggle/input', '/mnt/data']:
    p0=Path(root)
    if not p0.exists(): continue
    for p in p0.rglob('*'):
        if not p.is_file(): continue
        if p.suffix.lower()=='.dat': files.append((str(p),p.read_bytes()))
        elif p.suffix.lower()=='.zip':
            try: files.extend(nested_dat(p.read_bytes()))
            except (zipfile.BadZipFile, ValueError): pass
if not files:
    import requests
    for url in ['https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip',
                'https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip']:
        try:
            r=requests.get(url,timeout=90);r.raise_for_status()
            files=nested_dat(r.content)
            if files: break
        except Exception as e: print('Download attempt:',str(e)[:100])
unique={}
for name,blob in files:
    m=re.search(r'batch[_\s-]*(\d+)',name,re.I)
    if m: unique[int(m.group(1))]=(name,blob)
if len(unique)!=10: raise RuntimeError('Need all 10 UCI batch*.dat files. Attach official UCI Dataset.zip to Kaggle or enable Internet. Found: '+str(sorted(unique)))
X=[];Y=[];B=[]
for b,(name,blob) in sorted(unique.items()):
    for line in blob.decode('utf-8',errors='replace').splitlines():
        fields=line.split()
        if not fields: continue
        try:
            label=int(float(fields[0]));row=np.zeros(128,dtype=np.float64)
            for token in fields[1:]:
                if ':' not in token: continue
                i,v=token.split(':',1);i=int(i)-1
                if 0<=i<128: row[i]=float(v)
            X.append(row);Y.append(label);B.append(b)
        except (ValueError,IndexError): pass
X=np.asarray(X);Y=np.asarray(Y);B=np.asarray(B)
classes=np.sort(np.unique(Y)); assert len(classes)==6
labels=np.searchsorted(classes,Y)
source_ids=np.where(B<=7)[0];val_ids=np.where(B==8)[0];test_ids=np.where(B>=9)[0]
print('Dataset:',X.shape,'| source',len(source_ids),'validation',len(val_ids),'test',len(test_ids))

# ==================== ALIGNMENT ====================
def sample_ids(ids,n,seed):
    if n is None or len(ids)<=n: return np.sort(ids)
    try: return np.sort(train_test_split(ids,train_size=n,random_state=seed,stratify=labels[ids])[0])
    except ValueError: return np.sort(np.random.default_rng(seed).choice(ids,n,replace=False))

def sym_power(A,p):
    w,v=eigh((A+A.T)/2)
    return (v*np.maximum(w,1e-9)**p)@v.T

def coral(src,tgt,ridge=RIDGE):
    ms=src.mean(0);mt=tgt.mean(0)
    cs=np.cov(src,rowvar=False)+ridge*np.eye(src.shape[1])
    ct=np.cov(tgt,rowvar=False)+ridge*np.eye(src.shape[1])
    return (src-ms)@sym_power(cs,-0.5)@sym_power(ct,0.5)+mt

def mmd_align(src,tgt,seed):
    # Minimize multi-bandwidth RBF-MMD by learning a regularized affine
    # source transform; target labels are not used. Requires torch.
    import torch
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    if len(tgt)>MMD_FIT_TARGET_MAX:
        tgt=tgt[rng.choice(len(tgt),MMD_FIT_TARGET_MAX,replace=False)]
    if len(src)>MMD_FIT_TARGET_MAX:
        src_fit=src[rng.choice(len(src),MMD_FIT_TARGET_MAX,replace=False)]
    else: src_fit=src
    # Scale distances to stabilize MMD across domains
    scale=max(float(np.median(np.linalg.norm(src_fit[:min(150,len(src_fit))]-src_fit[0],axis=1))),1e-2)
    a=torch.tensor(src_fit/scale,dtype=torch.float32)
    b=torch.tensor(tgt/scale,dtype=torch.float32)
    d=src.shape[1]
    W=torch.nn.Parameter(torch.eye(d)); bias=torch.nn.Parameter(torch.zeros(d))
    opt=torch.optim.Adam([W,bias],lr=MMD_LR)
    def kern(x,z):
        dist=torch.cdist(x,z).square()
        return sum(torch.exp(-dist/(2*s*s)) for s in (0.5,1.,2.,4.))/4
    # Fixed deterministic 160-point subsets reduce MMD optimization cost
    ai=torch.randperm(len(a))[:min(160,len(a))]
    bi=torch.randperm(len(b))[:min(160,len(b))]
    aa=a[ai];bb=b[bi]
    for _ in range(MMD_STEPS):
        opt.zero_grad(); aa2=aa@W+bias
        loss=kern(aa2,aa2).mean()+kern(bb,bb).mean()-2*kern(aa2,bb).mean()
        loss=loss+MMD_REG*((W-torch.eye(d)).square().mean()+bias.square().mean())
        loss.backward();opt.step()
    return ((src/scale)@W.detach().numpy()+bias.detach().numpy())*scale

def adapt(src,tgt,method,seed):
    if method=='None': return src.copy()
    if method=='CORAL': return coral(src,tgt)
    if method=='MMD': return mmd_align(src,tgt,seed)
    raise ValueError(method)

# ==================== QUANTUM FEATURE MAP ====================
# PennyLane pure-state simulation. Squared inner products define a PSD quantum kernel.
try: import pennylane as qml
except ImportError:
    import sys,subprocess
    subprocess.check_call([sys.executable,'-m','pip','install','-q','pennylane'])
    import pennylane as qml
qdev=qml.device('default.qubit',wires=QUBITS)
@qml.qnode(qdev)
def quantum_state(x):
    for depth in range(Q_DEPTH):
        for w in range(QUBITS):
            qml.RY(x[w]*(depth+1),wires=w)
            qml.RZ(x[w]*0.5,wires=w)
        for w in range(QUBITS-1): qml.CNOT(wires=[w,w+1])
    return qml.state()

def q_states(arr):
    # bounded angles; fixed map, no training or test-label information
    angles=2*np.arctan(arr)
    return np.asarray([quantum_state(v) for v in angles],dtype=np.complex128)

def qkernel(A,B):
    return np.abs(A@B.conj().T)**2

# ==================== EXPERIMENT ====================
rows=[];predictions=[];timings=[]
methods=['None','CORAL','MMD']
models=['SVM','QSVM']
for seed in SEEDS:
    print('\nSEED',seed,flush=True)
    rng=np.random.default_rng(seed)
    tr=sample_ids(source_ids,N_TRAIN,seed)
    va=sample_ids(val_ids,N_VAL,seed)
    te=sample_ids(test_ids,N_TEST,seed)
    scaler=StandardScaler().fit(X[tr])
    tr0=scaler.transform(X[tr]);va0=scaler.transform(X[va]);te0=scaler.transform(X[te])
    pca=PCA(n_components=QUBITS,random_state=seed).fit(tr0)
    tr0=pca.transform(tr0);va0=pca.transform(va0);te0=pca.transform(te0)
    # Select C using only batch-8 labels, with adaptation to unlabeled batch-8.
    bestC={}
    for method in methods:
        trv=adapt(tr0,va0,method,seed)
        state_train=q_states(trv);state_val=q_states(va0)
        Ktr=qkernel(state_train,state_train);Kva=qkernel(state_val,state_train)
        for model in models:
            score_best=-np.inf;chosen=C_GRID[0]
            for C in C_GRID:
                if model=='SVM':
                    clf=SVC(C=C,kernel='rbf',gamma='scale',class_weight='balanced').fit(trv,labels[tr])
                    pv=clf.predict(va0)
                else:
                    clf=SVC(C=C,kernel='precomputed',class_weight='balanced').fit(Ktr,labels[tr])
                    pv=clf.predict(Kva)
                score=f1_score(labels[va],pv,average='macro',zero_division=0)
                if score>score_best: score_best=score;chosen=C
            bestC[(method,model)]=chosen
            print(' ',method,model,'validation F1',round(score_best,4),'C',chosen,flush=True)
    # Final evaluation uses ALL target samples; alignment uses their unlabeled features.
    for method in methods:
        t0=time.time()
        trt=adapt(tr0,te0,method,seed)
        qs=q_states(trt);qt=q_states(te0)
        Ktrain=qkernel(qs,qs);Ktest=qkernel(qt,qs)
        for model in models:
            C=bestC[(method,model)]
            if model=='SVM':
                clf=SVC(C=C,kernel='rbf',gamma='scale',class_weight='balanced').fit(trt,labels[tr])
                predicted=clf.predict(te0)
            else:
                clf=SVC(C=C,kernel='precomputed',class_weight='balanced').fit(Ktrain,labels[tr])
                predicted=clf.predict(Ktest)
            record={'seed':seed,'alignment':method,'model':model,'C':C,
                    'n_train':len(tr),'n_val':len(va),'n_test':len(te),
                    'accuracy':accuracy_score(labels[te],predicted),
                    'macro_f1':f1_score(labels[te],predicted,average='macro',zero_division=0),
                    'runtime_shared_alignment_and_kernel_sec':time.time()-t0}
            rows.append(record)
            for idx,actual,pr in zip(te,labels[te],predicted):
                predictions.append({'seed':seed,'alignment':method,'model':model,'original_index':int(idx),
                                    'batch':int(B[idx]),'true':int(actual),'predicted':int(pr)})
            print('  TEST',method,model,'Acc',round(record['accuracy'],4),'F1',round(record['macro_f1'],4),flush=True)
            if seed==SEEDS[0]:
                cm=confusion_matrix(labels[te],predicted,labels=np.arange(6))
                fig,ax=plt.subplots(figsize=(7,6))
                ConfusionMatrixDisplay(cm,display_labels=[str(c) for c in classes]).plot(ax=ax,cmap='Blues',colorbar=False)
                ax.set_title(f'{model} + {method} | seed {seed} | target batches 9-10')
                fig.tight_layout();fig.savefig(OUTPUT/f'confusion_{model}_{method}.png',dpi=170);plt.close(fig)
        # persist after each alignment to reduce lost progress
        pd.DataFrame(rows).to_csv(OUTPUT/'all_seed_results.csv',index=False)
        pd.DataFrame(predictions).to_csv(OUTPUT/'predictions.csv',index=False)

results=pd.DataFrame(rows)
summary=results.groupby(['alignment','model'],as_index=False).agg(
    n_seeds=('macro_f1','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
    macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'))
summary.to_csv(OUTPUT/'summary.csv',index=False)
# Paired differences by seed; CI is descriptive, not evidence of significance by itself.
comparisons=[]
for model in models:
    p=results[results.model==model].pivot(index='seed',columns='alignment',values='macro_f1')
    for method in ['CORAL','MMD']:
        diff=(p[method]-p['None']).dropna().values
        n=len(diff);mean=float(np.mean(diff));sd=float(np.std(diff,ddof=1)) if n>1 else float('nan')
        half=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else float('nan')
        comparisons.append({'model':model,'contrast':f'{method} - None','n':n,'mean_delta_f1':mean,
                            'ci95_low':mean-half,'ci95_high':mean+half,
                            'wilcoxon_p_exploratory':float(wilcoxon(diff).pvalue) if n>=5 and np.any(diff) else np.nan})
pd.DataFrame(comparisons).to_csv(OUTPUT/'paired_differences.csv',index=False)
fig,ax=plt.subplots(figsize=(9,5))
summary['label']=summary['model']+' + '+summary['alignment']
ax.bar(summary['label'],summary['macro_f1_mean'],yerr=summary['macro_f1_sd'].fillna(0),capsize=4)
ax.set_ylim(0,1);ax.set_ylabel('Macro F1 (mean ± seed SD)');ax.tick_params(axis='x',rotation=35)
fig.tight_layout();fig.savefig(OUTPUT/'model_comparison.png',dpi=180);plt.close(fig)
config={'seeds':SEEDS,'qubits':QUBITS,'depth':Q_DEPTH,'train_samples':N_TRAIN,'validation_samples':N_VAL,
        'test_samples':N_TEST,'C_grid':C_GRID,'methods':methods,'train_batches':'1-7','validation_batch':8,
        'test_batches':'9-10','adaptation':'transductive: unlabeled target features available before inference',
        'note':'Five seeds represent sampling variability on a fixed batch partition; not five independent datasets. No target test labels used for tuning.',
        'mmd':'regularized affine alignment optimized by Gaussian-kernel MMD; not MMD classifier loss',
        'quantum':'PennyLane CPU statevector squared-fidelity kernel; no hardware speedup claims'}
(OUTPUT/'experiment_config.json').write_text(json.dumps(config,indent=2))
zip_path=OUTPUT.parent/'Q2_Stage3_Research.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in OUTPUT.rglob('*'):
        if f.is_file():z.write(f,f.relative_to(OUTPUT))
print('\nFINAL SUMMARY\n',summary.to_string(index=False))
print('\nSaved:',zip_path)
try:
    from IPython.display import display,FileLink
    display(FileLink(str(zip_path)))
except ImportError: pass
