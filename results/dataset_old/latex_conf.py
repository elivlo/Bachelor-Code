

def build_latex_confusion_matrix(confusion_matrix):
    
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[1][0]
    tn = confusion_matrix[0][0]
    fn = confusion_matrix[0][1]
    
    m = "\def \cTP {%d}\n" %  tp
    m += "\def \cFP {%d}\n" % fp
    m += "\def \cTN {%d}\n" % tn
    m += "\def \cFN {%d}\n" % fn
    
    m += "\def \ctiP {%d}\n" % confusion_matrix["All"][1]
    m += "\def \ctiN {%d}\n" % confusion_matrix["All"][0]
    m += "\def \ctpP {%d}\n" % confusion_matrix[1]["All"]
    m += "\def \cFpN {%d}\n" % confusion_matrix[0]["All"]
    m += "\def \ctotal {%d}\n" % confusion_matrix["All"]["All"]
    
    m += "\def \cPPV {%.1f\\%%}\n" % (tp / (tp + fp) * 100)
    m += "\def \cTPR {%.1f\\%%}\n" % (tp / (tp + fn) * 100)
    m += "\def \cACC {%.1f\\%%}\n" % ((tp + tn) / (tp + fp + tn +fn) * 100)
    
    return m