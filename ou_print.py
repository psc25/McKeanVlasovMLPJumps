import numpy as np
import os

path = os.getcwd()
model = "ou"
dd = [10, 25, 50, 100, 200]
MNmax = 5
runs = 10
T = 1.0

# Table
txt = ""
for i in range(len(dd)):
    d = dd[i]
    txt = txt + str(dd[i]) + " & $L^2$-Error "
    for MN in range(1, MNmax+1):
        K = np.power(MN, MN)+1
        X_mlp = np.zeros([K, d, runs])
        X_tru = np.zeros([K, d, runs])
        try:
            for r in range(1, runs+1):
                X_mlp[:, :, r-1] = np.loadtxt(os.path.join(path, model, "mlp_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv"))
                X_tru[:, :, r-1] = np.loadtxt(os.path.join(path, model, "tru_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv"))
        except:
            print("Something is missing")
            
        X_mlp[X_mlp == 0.0] = np.nan
        print("d = " + str(d) + ", MN = " + str(MN) + ", loss = " + str(np.round(np.sqrt(np.nanmean(np.square(X_mlp - X_tru))), 4)))
        txt = txt + " & " + '{:.4f}'.format(np.round(np.sqrt(np.nanmean(np.square(X_mlp - X_tru))), 4))
        
    txt = txt + " \\\ \n"
    
    txt = txt + " & Time "
    for MN in range(1, MNmax+1):
        tms = np.zeros(runs)
        try:
            for r in range(1, runs+1):
                tms[r-1] = np.loadtxt(os.path.join(path, model, "tms_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv"))
        except:
            print("Something is missing")
            
        tms[tms==0.0] = np.nan
        t = np.nanmean(tms)
        if t < 0.0001:
            txt = txt + " & $<0.0001$"
        else:
            txt = txt + " & $" + '{:.4f}'.format(np.round(t, 4)) + "$ "
        
    txt = txt + " \\\ \n"
    
    txt = txt + " & Cost "
    for MN in range(1, MNmax+1):
        cst = np.zeros(runs)
        try:
            for r in range(1, runs+1):
                cst[r-1] = np.loadtxt(os.path.join(path, model, "cst_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv"))
        except:
            print("Something is missing")
            
        cst[tms==0.0] = np.nan
        t = np.nanmean(cst)
        
        if t == 0:
            txt = txt + " & "
        else:
            e = np.floor(np.log10(t))
            r = t/np.power(10, e)
            txt = txt + " & $" + '{:.2f}'.format(r) + " \\cdot 10^{" + '{:.0f}'.format(e) + "}$ "
        
    txt = txt + " \\\ \n\hline \n"
        
text_file = open(os.path.join(path, model + "_table.txt"), "w")
n = text_file.write(txt)
text_file.close()