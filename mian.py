from scr import CNN_Net, util, Train_Net, Test, LSTM_Net, \
    GRU_Net, GRU_Attention, CNN_16layers, \
    CNN_11layers, CNN_LSTM, Preprocessing, \
    SVM, Xgboost, Nature_ResNet

"""
Before run the function, you need download the dataset and rename the folder's name as mitbd. 
mit-bih: https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip
Download the Data, which is segment by an 1800 length of the sliding window, and the interval is 100: 
https://drive.google.com/file/d/1uce2k-iDb8Kd1tAYrvQlX7Hd1hsU-xno/view?usp=sharing

Put the download datas into the "Data" folder.

"""


def main():
    Data_path, Train_Path, Test_Path, Validate_Path, mitbd = util.GetDatapath()
    num = util.GetMethodIndex()
    use_gpu = util.gpu_available()
    if num == 0:
        data = Preprocessing.pp.LoadAllSignal(mitbd)
        label = Preprocessing.pp.LoadAllLabel(mitbd)
        R_detection = Preprocessing.pp.R_detection(data, 310)
        N_Seg, A_Seg, V_Seg, R_Seg, P_Seg, L_Seg = Preprocessing.pp.HeartBeatSegment(data, label, R_detection, 300)
        Data, Label, AllData = Preprocessing.pp.getDataLabel(N_Seg, A_Seg, V_Seg, R_Seg, P_Seg, L_Seg)
        AllTypeFeatureData, AllTypeFeatureLabel = Preprocessing.pp.FeatureExtraction(AllData)
        # Preprocessing.pp.Save(save_path, Data, Label) # change the save path
        # Preprocessing.pp.Save_feautre(save_path, AllTypeFeatureData, AllTypeFeatureLabel) # change the save path

    if num == 1:
        Feature, Label = SVM.SVM.LoadData(Data_path)
        train_x, test_x, train_y, test_y = SVM.SVM.Noramalization(Feature, Label)
        SVM_model = SVM.SVM.train_SVM(train_x, train_y)
        y_predict = SVM.SVM.Prediction(SVM_model, test_x, test_y)
        SVM.SVM.Test(y_predict, test_y)
        SVM.SVM.GetReport(test_y, y_predict)

    if num == 2:
        Feature, Label = Xgboost.Xgboost.LoadData(Data_path)
        train_x, test_x, train_y, test_y = Xgboost.Xgboost.Noramalization(Feature, Label)
        Xgboost_model = Xgboost.Xgboost.Train_Xgboost(train_x, train_y)
        y_predict = Xgboost.Xgboost.Prediction(Xgboost_model, test_x, test_y)
        Xgboost.Xgboost.Test(y_predict, test_y)
        Xgboost.Xgboost.GetReport(test_y, y_predict)

    if num == 3:
        train_loader, test_loader, test_data, test_label = util.ConstructHeartBeatData(Data_path, 32, 0.7, 7)

        cnn, optimizer, criterion = util.buildCNN(CNN_Net.CNN, 0.0001, use_gpu)
        print('Training Network....')
        Train_Net.Train(25, cnn, use_gpu, criterion, optimizer, train_loader, test_loader)
        Test.Test(cnn, test_data, test_label, use_gpu)

    if num == 4:
        train_loader, test_loader, test_data, test_label = util.ConstructHeartBeatData(Data_path, 16, 0.7)

        lstm, optimizer, criterion = util.buildLSTM(LSTM_Net.LSTM, 300, 150, 0.0005, use_gpu)

        Train_Net.Train(25, lstm, use_gpu, criterion, optimizer, train_loader, test_loader)
        Test.Test(lstm, test_data, test_label, use_gpu)

    if num == 5:
        train_loader, test_loader, validate_loader, test_data, test_label = util.ConstructSegData(Train_Path, Test_Path,
                                                                                                  Validate_Path,
                                                                                                  batch_size=32,
                                                                                                  num_workers=7,
                                                                                                  z_scale=False,
                                                                                                  gpu=use_gpu)
        lstm, optimizer, criterion = util.buildLSTM(LSTM_Net.LSTM, 1800, 256, 0.0005, use_gpu)
        print('Training Network....')
        Train_Net.Train(25, lstm, use_gpu, criterion, optimizer, train_loader, test_loader)
        Test.evaluation(lstm, validate_loader)

    if num == 6:
        train_loader, test_loader, validate_loader, test_data, test_label = util.ConstructSegData(Train_Path, Test_Path,
                                                                                                  Validate_Path,
                                                                                                  batch_size=64,
                                                                                                  num_workers=7,
                                                                                                  z_scale=False,
                                                                                                  gpu=use_gpu)
        gru, optimizer, criterion = util.buildGRU(GRU_Net.GRU, 1800, 600, 6, 0.0005, use_gpu)
        print('Training Network....')
        Train_Net.Train(30, gru, use_gpu, criterion, optimizer, train_loader, test_loader)
        Test.evaluation(gru, validate_loader)

    if num == 7:
        train_loader, test_loader, validate_loader, test_data, test_label = util.ConstructSegData(Train_Path, Test_Path,
                                                                                                  Validate_Path,
                                                                                                  batch_size=64,
                                                                                                  num_workers=7,
                                                                                                  z_scale=False,
                                                                                                  gpu=use_gpu)

        encoder, decoder, encoder_optimizer, decoder_optimizer, criterion = util.buildGRU_Attention(
            GRU_Attention.Encoder,
            GRU_Attention.AttnDecoderRNN,
            1800, 512, 0.0001, use_gpu)
        print('Training Network....')
        Train_Net.Train_GRUAttention(20, encoder, decoder,
                                     use_gpu, criterion, encoder_optimizer,
                                     decoder_optimizer, train_loader, test_loader)
        Test.gru_evaluation(encoder, decoder, validate_loader)

    if num == 8:
        print('Data resampling...')
        train_loader, test_loader, validate_loader, test_data, test_label = util.ConstructResample(Train_Path,
                                                                                                   Test_Path,
                                                                                                   Validate_Path,
                                                                                                   resample_size=1250,
                                                                                                   batch_size=64,
                                                                                                   num_workers=7,
                                                                                                   gpu=use_gpu)

        cnn, optimizer, criterion = util.buildCNN(CNN_11layers.CNN_11, 0.0001, use_gpu)
        print('Training Network....')
        Train_Net.Train(50, cnn, use_gpu, criterion, optimizer, train_loader, test_loader)
        Test.evaluation(cnn, validate_loader)

    if num == 9:
        train_loader, test_loader, validate_loader, test_data, test_label = util.ConstructSegData(Train_Path, Test_Path,
                                                                                                  Validate_Path,
                                                                                                  batch_size=64,
                                                                                                  num_workers=7,
                                                                                                  z_scale=False,
                                                                                                  gpu=use_gpu)
        cnn, optimizer, criterion = util.buildCNN(CNN_16layers.CNN_16, 0.0001, use_gpu)
        print('Training Network....')
        Train_Net.Train(20, cnn, use_gpu, criterion, optimizer, train_loader, test_loader)
        Test.evaluation(cnn, validate_loader)

    if num == 10:
        print('Data resampling...')
        train_loader, test_loader, validate_loader, test_data, test_label = util.ConstructResample(Train_Path,
                                                                                                   Test_Path,
                                                                                                   Validate_Path,
                                                                                                   resample_size=1000,
                                                                                                   batch_size=64,
                                                                                                   num_workers=7,
                                                                                                   gpu=use_gpu)
        cnn, optimizer, criterion = util.buildCNN(CNN_LSTM.CNN_LSTM, 0.0001, use_gpu)
        print('Training Network...')
        Train_Net.Train(150, cnn, use_gpu, criterion, optimizer, train_loader, test_loader)
        Test.evaluation(cnn, validate_loader)

    if num == 11:
        train_loader, test_loader, validate_loader, test_data, test_label = util.ConstructSegData(Train_Path, Test_Path,
                                                                                                  Validate_Path,
                                                                                                  batch_size=64,
                                                                                                  num_workers=7,
                                                                                                  z_scale=False,
                                                                                                  gpu=use_gpu)
        cnn, optimizer, criterion = util.buildCNN(Nature_ResNet.ResNet, 0.0001, use_gpu)
        print('Training Network...')
        Train_Net.Train(5, cnn, use_gpu, criterion, optimizer,train_loader, test_loader)
        Test.evaluation(cnn, validate_loader)



main()
