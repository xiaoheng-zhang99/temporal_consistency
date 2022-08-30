class hparams:
    seqlength = 855     # Maximum sequence length (sequences longer than this are dropped)
    IN_DIM = 512        # Feature dimension
    SEQ_DIM = 128
    HIDDEN_DIM = 64
    ATTEN_SIZE = 16
    keep_proba = 0.9
    weight_decay = 1e-3
    #train hyper-parameters
    BATCH_SIZE = 16
    num_train_steps = 3000
    lr = 1e-4
    #data_file
    emo_train_file = 'emo_train.csv'
    emo_test_file = 'emo_test.csv'
    #model_save
    model_path_save='./model/model'
    model_path_load='./model/model'
    val_set = 'Ses01'
    '''
    import argparse


class Hparame:
    parser = argparse.ArgumentParser()

    # data preprocessing
    parser.add_argument('--DATA_COLUMN', default="sentence", help="data column")
    parser.add_argument('--LABEL_COLUMN', default="polarity", help="polarity")
    parser.add_argument('--label_list', default="0,1", help="label_list ")

    # This is a path to an uncased (all lowercase) version of BERT
    # parser.add_argument("--BERT_MODEL_HUB", default="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
    parser.add_argument("--BERT_INIT_CHKPNT", default="./bert_pretrain_model/bert_model.ckpt")
    parser.add_argument("--BERT_VOCAB", default="./bert_pretrain_model/vocab.txt")
    parser.add_argument("--BERT_CONFIG", default="./bert_pretrain_model/bert_config.json")

    # We'll set sequences to be at most 128 tokens long.
    parser.add_argument("--MAX_SEQ_LENGTH", default=128, type=int)

    """ train hyper-parameters """
    parser.add_argument("--BATCH_SIZE", default=32, type=int)
    parser.add_argument("--LEARNING_RATE", default=2e-5, type=float)
    parser.add_argument("--NUM_TRAIN_EPOCHS", default=3.0, type=float)

    # Warmup is a period of time where hte learning rate
    # is small and gradually increases--usually helps training.
    parser.add_argument("--WARMUP_PROPORTION", default=0.1, type=float)

    # Model configs
    parser.add_argument("--SAVE_CHECKPOINTS_STEPS", default=500, type=int)
    parser.add_argument("--SAVE_SUMMARY_STEPS", default=100, type=int)

    """ save model """
    parser.add_argument("--OUTPUT_DIR", default="./save_model/")
    parser.add_argument("--model_output", default="bert_model")
    '''