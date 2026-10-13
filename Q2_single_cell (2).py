# Q2: Drift-Robust Quantum Machine Learning for Electronic Nose Sensor Classification
# One Kaggle cell. Attach UCI Gas Sensor Array Drift ZIP or enable Internet.
import os, sys, subprocess, pathlib, zipfile, io, re, json, time, random, warnings
warnings.filterwarnings('ignore')
for mod,pkg in [('pennylane','pennylane'),('requests','requests')]:
 try: __import__(mod)
 except ImportError: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])
import numpy as np, pandas as pd, torch, torch.nn as nn, pennylane as qml, requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pathlib import Path
SEED=42; N_QUBITS=4; LAYERS=2; QSVM_TRAIN=240; TEST_CAP=240; VQC_TRAIN=500; VQC_EPOCHS=12; BATCH_SIZE=16
OUT=Path('/kaggle/working/Q2_Drift_Robust_QML'); OUT.mkdir(parents=True,exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print('PyTorch:',torch.__version__,'PennyLane:',qml.__version__,'CUDA available:',torch.cuda.is_available(),'(quantum simulator uses CPU)')
def nested_zip(data,depth=0):
 if depth>4:return []
 result=[]
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  for name in z.namelist():
   if name.lower().endswith('.dat'):result.append((name,z.read(name)))
   elif name.lower().endswith('.zip'):result+=nested_zip(z.read(name),depth+1)
 return result
files=[]
for p in Path('/kaggle/input').rglob('*'):
 if not p.is_file():continue
 try:
  if p.suffix.lower()=='.dat':files.append((str(p),p.read_bytes()))
  elif p.suffix.lower()=='.zip':files+=nested_zip(p.read_bytes())
 except Exception:pass
if not files:
 for url in ['https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip','https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip']:
  try:
   r=requests.get(url,timeout=90);r.raise_for_status();files=nested_zip(r.content)
   if files:break
  except Exception as e:print('Download attempt:',str(e)[:100])
if not files:raise RuntimeError('Attach official UCI Gas Sensor Array Drift dataset ZIP to Kaggle or enable Internet.')
rows=[]; labels=[]; batches=[]; seen=set()
for name,blob in files:
 match=re.search(r'batch[_\s-]*(\d+)',name,re.I)
 if not match:continue
 b=int(match.group(1))
 # Ignore duplicate copies of the same batch file
 if b in seen:continue
 seen.add(b)
 for line in blob.decode('utf-8',errors='replace').splitlines():
  t=line.strip().split()
  if len(t)<2:continue
  try:
   v=np.zeros(128,dtype=np.float32)
   for term in t[1:]:
    if ':' not in term:continue
    idx,val=term.split(':',1);idx=int(idx)-1
    if 0<=idx<128:v[idx]=float(val)
   rows.append(v);labels.append(int(float(t[0])));batches.append(b)
  except (ValueError,IndexError):pass
X=np.asarray(rows,dtype=np.float32);yraw=np.asarray(labels);batch=np.asarray(batches)
assert len(X)>0 and len(np.unique(yraw))==6,'Check dataset files and labels'
classes=np.sort(np.unique(yraw));y=np.searchsorted(classes,yraw)
print('Loaded:',X.shape,'batches:',sorted(set(batch)),'class counts:',np.bincount(y))
assert all(np.isin([1,2,3,4,5,6,7,8,9,10],np.unique(batch))),'All 10 batch files are required.'
# Fixed chronological evaluation: train 1-7, validation 8, test 9-10
train_idx=np.where(batch<=7)[0];val_idx=np.where(batch==8)[0];test_idx=np.where(batch>=9)[0]
# Subset selection is reproducible; stratify when possible.
def cap(idx,n):
 if len(idx)<=n:return idx
 try:a,_=train_test_split(idx,train_size=n,stratify=y[idx],random_state=SEED);return np.sort(a)
 except ValueError:return np.sort(np.random.default_rng(SEED).choice(idx,n,replace=False))
qtrain=cap(train_idx,QSVM_TRAIN);vtrain=cap(train_idx,VQC_TRAIN);qtest=cap(test_idx,TEST_CAP)
# Preprocessing learned on chronological training ONLY; same 4-D features for all models.
imp=SimpleImputer(strategy='median');sc=StandardScaler();pca=PCA(n_components=N_QUBITS,random_state=SEED)
A=imp.fit_transform(X[train_idx]);A=sc.fit_transform(A);pca.fit(A)
def features(idx):return pca.transform(sc.transform(imp.transform(X[idx]))).astype('float32')
# Bound angles for numerical stability; parameters fitted on train only.
Ftrain=features(train_idx);scale=np.maximum(np.quantile(np.abs(Ftrain),0.95,axis=0),1e-6)
def angles(idx):return (np.pi*np.tanh(features(idx)/scale)).astype('float32')
Xt=angles(qtrain);Xv=angles(val_idx);Xe=angles(qtest)
records=[];predictions=[]
def record(model,truth,pred,secs,idx):
 a=accuracy_score(truth,pred);f=f1_score(truth,pred,average='macro',zero_division=0)
 records.append(dict(model=model,accuracy=a,macro_f1=f,seconds=secs,n_train=QSVM_TRAIN if model!='VQC' else len(vtrain),n_test=len(truth)))
 predictions.extend(dict(model=model,index=int(i),batch=int(batch[i]),actual=int(t),predicted=int(p)) for i,t,p in zip(idx,truth,pred))
 print(f'{model}: accuracy={a:.4f}, macro_F1={f:.4f}, time={secs:.1f}s')
 (OUT/f'{model}_report.txt').write_text(classification_report(truth,pred,labels=np.arange(6),zero_division=0))
 fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay(confusion_matrix(truth,pred,labels=np.arange(6))).plot(ax=ax,cmap='Blues',colorbar=False)
 ax.set_title(model+' chronological drift');fig.tight_layout();fig.savefig(OUT/f'{model}_confusion.png',dpi=160);plt.close(fig)
# Classical SVM matched to QSVM subset
start=time.time();svm=SVC(C=1,kernel='rbf',gamma='scale');svm.fit(Xt,y[qtrain]);pred=svm.predict(Xe)
record('SVM',y[qtest],pred,time.time()-start,qtest)
# QSVM quantum fidelity kernel: pairwise state overlaps. State vectors cached once per sample.
qdev=qml.device('default.qubit',wires=N_QUBITS)
@qml.qnode(qdev)
def state(x):
 qml.AngleEmbedding(x,wires=range(N_QUBITS),rotation='Y')
 for i in range(N_QUBITS):qml.CNOT(wires=[i,(i+1)%N_QUBITS])
 qml.AngleEmbedding(x,wires=range(N_QUBITS),rotation='Z')
 return qml.state()
start=time.time();Str=np.asarray([state(z) for z in Xt]);Ste=np.asarray([state(z) for z in Xe])
Ktr=np.clip(np.abs(Str@Str.conj().T)**2,0,1);Kte=np.clip(np.abs(Ste@Str.conj().T)**2,0,1)
qsvm=SVC(C=1,kernel='precomputed');qsvm.fit(Ktr,y[qtrain]);pred=qsvm.predict(Kte)
record('QSVM',y[qtest],pred,time.time()-start,qtest)
# VQC, trained only on batch 1-7, tuned by validation batch 8
qdev2=qml.device('default.qubit',wires=N_QUBITS)
@qml.qnode(qdev2,interface='torch',diff_method='backprop')
def circuit(inputs,weights):
 qml.AngleEmbedding(inputs,wires=range(N_QUBITS),rotation='Y')
 qml.StronglyEntanglingLayers(weights,wires=range(N_QUBITS))
 return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
class VQC(nn.Module):
 def __init__(self):
  super().__init__();self.quantum=qml.qnn.TorchLayer(circuit,{'weights':(LAYERS,N_QUBITS,3)});self.head=nn.Linear(N_QUBITS,6)
 def forward(self,x):return self.head(self.quantum(x).float())
model=VQC();opt=torch.optim.Adam(model.parameters(),lr=.02);loss_fn=nn.CrossEntropyLoss()
Xvt=torch.tensor(angles(vtrain));Yvt=torch.tensor(y[vtrain],dtype=torch.long)
Xval=torch.tensor(angles(val_idx));Yval=torch.tensor(y[val_idx],dtype=torch.long)
loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xvt,Yvt),batch_size=BATCH_SIZE,shuffle=True)
best=-1;best_epoch=0;history=[];start=time.time();checkpoint=OUT/'VQC_best.pt'
for epoch in range(VQC_EPOCHS):
 model.train();total=0
 for xx,yy in loader:
  opt.zero_grad();logits=model(xx);loss=loss_fn(logits,yy);loss.backward();opt.step();total+=loss.item()*len(xx)
 model.eval()
 with torch.no_grad():vp=np.concatenate([model(z).argmax(1).numpy() for z in Xval.split(32)])
 vf=f1_score(y[val_idx],vp,average='macro',zero_division=0);history.append(dict(epoch=epoch+1,train_loss=total/len(vtrain),validation_macro_f1=vf))
 print(f'VQC epoch {epoch+1}: validation macro F1={vf:.4f}')
 if vf>best+1e-5:best=vf;best_epoch=epoch+1;torch.save(model.state_dict(),checkpoint)
 if epoch+1-best_epoch>=4:break
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
with torch.no_grad():pred=np.concatenate([model(z).argmax(1).numpy() for z in torch.tensor(Xe).split(32)])
record('VQC',y[qtest],pred,time.time()-start,qtest)
pd.DataFrame(records).to_csv(OUT/'model_comparison.csv',index=False)
pd.DataFrame(predictions).to_csv(OUT/'test_predictions.csv',index=False)
pd.DataFrame(history).to_csv(OUT/'vqc_learning_curve.csv',index=False)
pd.DataFrame({'sample_index':np.r_[qtrain,val_idx,qtest],'partition':['train']*len(qtrain)+['validation']*len(val_idx)+['test']*len(qtest)}).to_csv(OUT/'split_manifest.csv',index=False)
with open(OUT/'configuration.json','w') as f:json.dump(dict(seed=SEED,qubits=N_QUBITS,layers=LAYERS,qsvm_train=len(qtrain),vqc_train=len(vtrain),test=len(qtest),validation=len(val_idx),train_batches='1-7',validation_batch=8,test_batches='9-10',note='SVM and QSVM matched training subset; VQC has a larger training subset. All use same test samples.'),f,indent=2)
fig,ax=plt.subplots(figsize=(7,4));df=pd.DataFrame(records);ax.bar(df.model,df.macro_f1);ax.set_ylim(0,1);ax.set_ylabel('Macro F1');ax.set_title('Q2: Chronological batch generalization');fig.tight_layout();fig.savefig(OUT/'model_comparison.png',dpi=160);plt.close(fig)
archive=Path('/kaggle/working/Q2_Drift_Robust_QML.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(OUT))
print('\nRESULTS\n',df.to_string(index=False));print('ZIP:',archive)
try:
 from IPython.display import display,FileLink
 display(FileLink(str(archive)))
except Exception:pass
