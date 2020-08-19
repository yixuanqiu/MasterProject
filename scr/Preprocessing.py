import glob
from biosppy.signals import ecg
import pywt
import wfdb
import numpy as np


class pp():

    @staticmethod
    def Denoise(sig):
        """
        Wavelet denoise
        """
        coeffs = pywt.wavedec(sig, 'db6', level=9)
        coeffs[-1] = np.zeros(len(coeffs[-1]))
        coeffs[-2] = np.zeros(len(coeffs[-2]))
        coeffs[0] = np.zeros(len(coeffs[0]))
        sig_filt = pywt.waverec(coeffs, 'db6')
        return sig_filt
    

    def LoadAllSignal(data_path):
        print("Loading data...")
        Signalsample = []
        file0 = glob.glob(data_path + '*.hea')
        for i in range(len(file0)):
            annotation = wfdb.rdann(data_path + file0[i][-7:-4], 'atr')
            record_name = annotation.record_name  # read the record name
            signal = wfdb.rdsamp(data_path + record_name)[0][:, 0]  # get the data of lead0 'MILLI'
            Signalsample.append(signal)            
        return Signalsample
    
    
    @staticmethod
    def LoadAllLabel(data_path):
        print("Loading Label...")
        LabelSample = []
        file0 = glob.glob(data_path + '*.hea')
        for i in range(len(file0)):
            annotation = wfdb.rdann(data_path + file0[i][-7:-4], 'atr')
            Label = annotation.symbol
            LabelSample.append(Label)            
        return LabelSample
    
    
    @staticmethod
    def R_detection(Signal, fs):   
        """
        Input - FilteredSignal: after denoising
                fs            : sampling frequency
        Output - index of R-peak
        """
        print("R peak detecting...")
        RPeakIndex = []
        for i in range(len(Signal)):
            sig = Signal[i]
            record = pp.Denoise(sig)
            rpeaks = ecg.christov_segmenter(record, sampling_rate=fs)
            rpeaks = rpeaks[0]
            RPeakIndex.append(rpeaks)
        return RPeakIndex


    @staticmethod
    def HeartBeatSegment(Signalsample, LabelSample, RPeakIndex,fs):
        N_Seg = []
        A_Seg = []
        V_Seg = []
        R_Seg = []
        P_Seg = []
        L_Seg = []
        for i in range(len(Signalsample)):
            record = Signalsample[i]
            label_index = RPeakIndex[i]
            label = LabelSample[i]
            for j in range(len(label_index)):
                if label_index[j] >= 120 and (label_index[j] + 180) <= 650000:
                    if label[j] == 'N' or '·':
                            Seg = record[label_index[j] - 120:label_index[j] + 180]  # before r peak fs*40%, after R peak fs*60%
                            N_Seg.append(Seg)
        
                    if label[j] == 'A':
                            Seg = record[label_index[j] - 120:label_index[j] + 180]
                            A_Seg.append(Seg)
        
                    if label[j] == 'V':
                            Seg = record[label_index[j] - 120:label_index[j] + 180]
                            V_Seg.append(Seg)
        
                    if label[j] == 'R':
                            Seg = record[label_index[j] - 120:label_index[j] + 180]
                            R_Seg.append(Seg)
                    if label[j] == '/':
                            Seg = record[label_index[j] - 120:label_index[j] + 180]
                            P_Seg.append(Seg)
                    if label[j] == 'L':
                            Seg = record[label_index[j] - 120:label_index[j] + 180]
                            L_Seg.append(Seg)
        return N_Seg, A_Seg, V_Seg, R_Seg, P_Seg, L_Seg
    
    @staticmethod
    def getDataLabel(N_Seg, A_Seg, V_Seg, R_Seg, P_Seg, L_Seg):
        AllData = []
        N_segment = np.array(N_Seg)
        N_segment = N_segment[0:7000, :]
        AllData.append(N_segment)
        A_segment = np.array(A_Seg)
        AllData.append(A_segment)
        V_segment = np.array(V_Seg)
        R_segment = np.array(R_Seg)
        P_segment = np.array(P_Seg)
        L_segment = np.array(L_Seg)
        AllData.append(V_segment)
        AllData.append(R_segment)
        AllData.append(P_segment)
        AllData.append(L_segment)

        label_N = np.zeros(N_segment.shape[0])
        label_A = np.ones(A_segment.shape[0])
        label_V = np.ones(V_segment.shape[0]) * 2
        label_R = np.ones(R_segment.shape[0]) * 3
        label_P = np.ones(P_segment.shape[0]) * 4
        label_L = np.ones(L_segment.shape[0]) * 5
        
        Data = np.concatenate((N_segment, A_segment, V_segment, R_segment, P_segment, L_segment), axis=0)
        Label = np.concatenate((label_N, label_A, label_V, label_R, label_P, label_L), axis=0)
        

        return Data, Label, AllData
        
    @staticmethod
    def Save(save_path, Data, Label):
        np.save(save_path + 'Data', Data)
        np.save(save_path+ 'Label', Label)
    
    @staticmethod
    def WaveletAlternation(Segment):
        Feature_dim, SingleDir_Samples = Segment.shape    # get the shape of the segment
        SingleDir_SamplesFeature =np.zeros((Feature_dim,32)) # set a feature matrix
        for i in range(Feature_dim):
            SingleSampleDataWavelet = Segment[i,:] # Wavelet packet decomposition
            # feature extraction using Wavelet packet decomposition
            wp = pywt.WaveletPacket(SingleSampleDataWavelet, wavelet='db6', mode='symmetric', maxlevel=5) # use db6
        #            print([node.path for node in wp.get_leaf(3, 'natural')])
            # there is 32 nodes in the 5 level
            aaaaa = wp['aaaaa'].data  # the first node
            aaaad = wp['aaaad'].data  # the second node
            aaada = wp['aaada'].data
            aaadd = wp['aaadd'].data
            aadaa = wp['aadaa'].data
            aadad = wp['aadad'].data
            aadda = wp['aadda'].data
            aaddd = wp['aaddd'].data
            adaaa = wp['adaaa'].data
            adaad = wp['adaad'].data
            adada = wp['adada'].data
            adadd = wp['adadd'].data
            addaa = wp['addaa'].data
            addad = wp['addad'].data
            addda = wp['addda'].data
            adddd = wp['adddd'].data
            daaaa = wp['daaaa'].data
            daaad = wp['daaad'].data
            daada = wp['daada'].data
            daadd = wp['daadd'].data
            dadaa = wp['dadaa'].data
            dadad = wp['dadad'].data
            dadda = wp['dadda'].data
            daddd = wp['daddd'].data
            ddaaa = wp['ddaaa'].data
            ddaad = wp['ddaad'].data
            ddada = wp['ddada'].data
            ddadd = wp['ddadd'].data
            dddaa = wp['dddaa'].data
            dddad = wp['dddad'].data
            dddda = wp['dddda'].data
            ddddd = wp['ddddd'].data
            # calculate Norm
            # Squared sum of node coefficients obtained from the parametrization / matrix elements.
            ret1 = np.linalg.norm(aaaaa, ord=None)
            ret2 = np.linalg.norm(aaaad, ord=None)
            ret3 = np.linalg.norm(aaada, ord=None)
            ret4 = np.linalg.norm(aaadd, ord=None)
            ret5 = np.linalg.norm(aadaa, ord=None)
            ret6 = np.linalg.norm(aadad, ord=None)
            ret7 = np.linalg.norm(aadda, ord=None)
            ret8 = np.linalg.norm(aaddd, ord=None)
            ret9 = np.linalg.norm(adaaa, ord=None)
            ret10 = np.linalg.norm(adaad, ord=None)
            ret11 = np.linalg.norm(adada, ord=None)
            ret12 = np.linalg.norm(adadd, ord=None)
            ret13 = np.linalg.norm(addaa, ord=None)
            ret14 = np.linalg.norm(addad, ord=None)
            ret15 = np.linalg.norm(addda, ord=None)
            ret16 = np.linalg.norm(adddd, ord=None)
            ret17 = np.linalg.norm(daaaa, ord=None)
            ret18 = np.linalg.norm(daaad, ord=None)
            ret19 = np.linalg.norm(daada, ord=None)
            ret20 = np.linalg.norm(daadd, ord=None)
            ret21 = np.linalg.norm(dadaa, ord=None)
            ret22 = np.linalg.norm(dadad, ord=None)
            ret23 = np.linalg.norm(dadda, ord=None)
            ret24 = np.linalg.norm(daddd, ord=None)
            ret25 = np.linalg.norm(ddaaa, ord=None)
            ret26 = np.linalg.norm(ddaad, ord=None)
            ret27 = np.linalg.norm(ddada, ord=None)
            ret28 = np.linalg.norm(ddadd, ord=None)
            ret29 = np.linalg.norm(dddaa, ord=None)
            ret30 = np.linalg.norm(dddad, ord=None)
            ret31 = np.linalg.norm(dddda, ord=None)
            ret32 = np.linalg.norm(ddddd, ord=None)
            # 32 nodes combine to form a feature vector
            SingleSampleFeature = [ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8, ret9, ret10, ret11, ret12, ret13
                , ret14, ret15, ret16, ret17, ret18, ret19, ret20, ret21, ret22, ret23, ret24, ret25
                , ret26, ret27, ret28, ret29, ret30, ret31, ret32]
            SingleDir_SamplesFeature[i][:] = SingleSampleFeature # get array
        return SingleDir_SamplesFeature
    
    
    
    @staticmethod
    def FeatureExtraction(AllData):
        AllTypeFeatureData = []
        AllTypeFeatureLabel =[]
        print("Feature extraction...")

        for i in range(len(AllData)):
            signal = AllData[i]
            Results = pp.WaveletAlternation(signal)
            AllTypeFeatureData.append(Results)
            label = np.ones(Results.shape[0])*i
            AllTypeFeatureLabel.append(label)
        AllTypeFeatureData = np.array(AllTypeFeatureData)
        AllTypeFeatureLabel = np.array(AllTypeFeatureLabel)
        
        return AllTypeFeatureData, AllTypeFeatureLabel
    
    @staticmethod
    def Save_feautre(save_path, AllTypeFeatureData, AllTypeFeatureLabel):
        np.save(save_path + 'feature_Data', AllTypeFeatureData)
        np.save(save_path + 'feature_Label', AllTypeFeatureLabel)
        print('Feature Data and Feature Label have been saved!')
    
        
         
      
    
    
    


# data_path = '/Users/andyq/Documents/Code/ECGproject/MIT_BIH/'
# A = Preprocessing.LoadAllSignal(data_path)
# B = Preprocessing.LoadAllLabel(data_path)
# R = Preprocessing.R_detection(A, 310)

# N_Seg, A_Seg, V_Seg, R_Seg, P_Seg, L_Seg = Preprocessing.HeartBeatSegment(A, B, R, 300)
# Data, Label, AllData = Preprocessing.getDataLabel(N_Seg, A_Seg, V_Seg, R_Seg, P_Seg, L_Seg)       
# AllTypeFeatureData, AllTypeFeatureLabel= Preprocessing.FeatureExtraction(AllData)


