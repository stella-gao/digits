import clean_digits

def load_data_set(train_speakers, test_speakers):

    X_train, y_train = clean_digits.load_speaker_set(train_speakers)
    X_test, y_test = clean_digits.load_speaker_set(test_speakers)
                        
    return ((X_train, y_train), (X_test, y_test))


train_speakers = [
        'MAE',
        'MBD',
        'MCB',
        'FAC',
        'FBH',
        'FCA',
        'MDL',
        'MEH',
        'FDC']
    
test_speakers = ['FEA']

(X_train_wav, y_train), (X_test_wav, y_test) = load_data_set(train_speakers, test_speakers)
print y_test


