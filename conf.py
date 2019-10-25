#定义超参数
#定义默认文件地址
corpus_path = "corpus"
model_path = "model"
data_path = "data"
#定义产参数
#输入词典大小
enc_vocab_size = 20000
#输出词典大小
dec_vocab_size = 20000
#最长输入字符长度
enc_input_length = 50
dec_output_length = 50
#输入的嵌入embdding size
enc_embedding_length = 128
dec_embedding_length = 128
#神经网络隐含层神经元数
hidden_dim = 100
#attentin layer shape 表示lstm深度，1是encode 2是decode
layer_shape = (2, 4)
#增加到方差的小的浮点数，以避免除以零
epsilon = 1e-6

#训练数据文件名
source="question.txt"
target="answer.txt"