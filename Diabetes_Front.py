from tkinter import *
from tkinter import ttk
import pandas as pd
import backone
import matplotlib
from matplotlib import style
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score,f1_score

f = Figure(figsize = (5,4),dpi = 100)
a = f.add_subplot(111)

f2 = Figure(figsize = (5,4),dpi = 100)
a2 = f2.add_subplot(111)

def plot_roc_curve(plt,fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc = 'upper left')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')

def plot_precision_recall_vs_threshold(plt,precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left",)
    plt.ylim([0, 1])
    return

def clearAnalysis(canvas):
    a.clear()
    canvas.show()
    canvas.get_tk_widget().pack(fill = BOTH ,expand=True)
    return

def callback(algo,frame2_1,canvas,text):
    a.set_xlabel('False Positive Rate')
    a.set_ylabel('True Posititve Rate')
    text.delete('1.0','8.0 lineend')
    fpr,tpr,thresholds = roc_curve(backone.trainY,backone.result2[algo.get()])
    plot_roc_curve(a,fpr,tpr)
    s = """Accuracy: {}% (+/- {}%) \n
        Precision is {} \n
        Recall is {} \n
        F1 Score is {}
        """.format( 100*backone.results[algo.get()].mean(), 100*backone.results[algo.get()].std() * 2
            , precision_score(backone.trainY,backone.result2[algo.get()]), recall_score(backone.trainY,backone.result2[algo.get()])
            , f1_score(backone.trainY, backone.result2[algo.get()]))
    text.insert("1.0",s)

    canvas.show()
    canvas.get_tk_widget().pack(fill = BOTH ,expand=True)
    return
data =  pd.read_csv('/home/atharva/Downloads/diabetes2.csv')

def plotGraph(Xvalue,Yvalue,canvas2):
    a2.clear()
    a2.scatter(data[Xvalue.get()],data[Yvalue.get()],c=data['Outcome'])
    a2.set_xlabel(Xvalue.get())
    a2.set_ylabel(Yvalue.get())
    a2.legend()
    canvas2.show()
    canvas2.get_tk_widget().pack(fill = BOTH,expand=True)
    return

def main():
    root = Tk()
    algotab = ttk.Notebook(root)
    #####################################################################################
    ############################# Visulaization Tab #####################################
    #####################################################################################
    panedwindowViz = ttk.Panedwindow(algotab,orient = HORIZONTAL)
    Xvalue = StringVar()
    Yvalue = StringVar()
    paraframe = ttk.Frame(panedwindowViz,width = 100, height= 300, relief=SUNKEN)
    xlabel = ttk.Label(paraframe,text ="X Variable")
    xlabel.pack(fill = BOTH,expand =True)
    xcombobox = ttk.Combobox(paraframe,textvariable = Xvalue,values = ('Pregnancies',"Glucose","BloodPressure",
    "SkinThickness",'Insulin','BMI','DiabetesPedigreeFunction','Age')).pack(fill =BOTH,expand=True)
    ylabel = ttk.Label(paraframe,text = "Y Variable")
    ylabel.pack(fill = BOTH,expand = True)
    ycombobox = ttk.Combobox(paraframe,textvariable = Yvalue,values = ('Pregnancies',"Glucose","BloodPressure",
    "SkinThickness",'Insulin','BMI','DiabetesPedigreeFunction','Age')).pack(fill =BOTH,expand=True)
    vizbutton = ttk.Button(paraframe,text = "Plot Graph",command = lambda: plotGraph(Xvalue,Yvalue,canvas2)).pack(fill =BOTH,expand =True)


    vizframe =  ttk.Frame(panedwindowViz,width = 600 , height = 300,relief=SUNKEN)
    canvas2 =  FigureCanvasTkAgg(f2,vizframe)
    toolbar2 = NavigationToolbar2TkAgg(canvas2, vizframe)
    toolbar2.update()
    canvas2._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
    panedwindowViz.add(paraframe,weight = 1)
    panedwindowViz.add(vizframe,weight = 4)
    panedwindowViz.pack(fill = BOTH,expand = True)
    algotab.add(panedwindowViz,text="Data Visualization")

    ####################################################
    ################# ALGORITHM TAB ####################
    #####################################################
    panedwindow = ttk.Panedwindow(algotab,orient =HORIZONTAL)
    panedwindow.pack(fill =BOTH,expand = True)
    algo =StringVar()
    frame1 = ttk.Frame(panedwindow,width =100,height = 300,relief =SUNKEN)
    combobox = ttk.Combobox(frame1,textvariable = algo,values = ('Nearest Neigbhors','Random Forest','Decision Tree',
    'Neural Net','Linear SVM','AdaBoost')).pack(fill =BOTH,expand = True)
    r1= ttk.Radiobutton(frame1,text = 'Random Forest',variable= algo,value ="Random Forest" ).pack(fill =BOTH,expand = True)
    r1= ttk.Radiobutton(frame1,text = 'Decision Tree',variable= algo,value ="Decision Tree" ).pack(fill =BOTH,expand = True)
    r1= ttk.Radiobutton(frame1,text = 'Nearest Neighbors',variable= algo,value ="Nearest Neighbors" ).pack(fill =BOTH,expand = True)
    r1= ttk.Radiobutton(frame1,text = 'Neural Net',variable= algo,value ="Neural Net" ).pack(fill =BOTH,expand = True)
    r1= ttk.Radiobutton(frame1,text = 'Linear SVM',variable= algo,value ="Linear SVM" ).pack(fill =BOTH,expand = True)
    r1= ttk.Radiobutton(frame1,text = 'AdaBoost',variable= algo,value ="AdaBoost" ).pack(fill =BOTH,expand = True)

    frame2= ttk.Frame(panedwindow,width =600,height = 300,relief =SUNKEN)

    panedwindow.add(frame1,weight=1)
    panedwindow.add(frame2,weight=4)

    panedwindow2 = ttk.Panedwindow(frame2,orient = VERTICAL)
    panedwindow2.pack(fill = BOTH, expand =True)
    algotab.add(panedwindow,text = "ALGORITHM")

    algotab.pack(fill = BOTH,expand = True)

    frame2_1 = ttk.Frame(panedwindow2,width = 600,height =400,relief=SUNKEN)
    frame2_2 = ttk.Frame(panedwindow2,width = 600,height = 100,relief=SUNKEN)
    canvas =  FigureCanvasTkAgg(f,frame2_1)
    button = ttk.Button(frame1,text = "Show Analysis",command = lambda: callback(algo,frame2_1,canvas,text)).pack(fill =BOTH,expand =True)
    button = ttk.Button(frame1,text = "Clear Analysis",command = lambda: clearAnalysis(canvas)).pack(fill = BOTH,expand =True)
    panedwindow2.add(frame2_1,weight=1)
    panedwindow2.add(frame2_2,weight=1)


    text = Text(frame2_2,width = 15,height = 7)
    text.pack(fill = BOTH,expand=True)
    toolbar = NavigationToolbar2TkAgg(canvas, frame2_1)
    toolbar.update()
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

    #####################################################################################
    ############################# END Tab #####################################
    #####################################################################################


    root.mainloop()

if __name__== '__main__':main()
