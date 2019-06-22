from django.shortcuts import render,render_to_response
import re,jieba,gensim,math                #jieba分词库
import numpy as np
import networkx as nx       #用于图的建立
import itertools

class tr(object):
    #切分句子
    def cut_sents(content):
        '''
        :param content:输入一段话
        :return: 返回该段话的句子分割
        '''
        sentences = re.split(r"([。!！?？\s+])", content)[:-1]       #按照中文的句子结束符号对一段话进行句子分割
        sentences.append("")
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        s_token = sentences[0].split("，")       #对每段首句额外使用“，”进行分割
        del sentences[0]
        sentences = s_token + sentences     #更新句子列表
        return sentences

    #切分单词
    def cut_word(context):
        '''
        :param context: 分句后的句子列表
        :return: total_cutword是对每个句子进行分词，去除停用词的结果，类型是列表的列表；total_content是过滤了全停用词组成的句子后的新的句子列表，与前者一一对应
        '''
        with open(r't_rank\stopwords.txt',encoding='utf-8') as f:
            stopkey = [line.strip() for line in f.readlines()]        #加载停用词表
        total_cutword = []      #每个句子的分词结果
        total_content = []      #存储不是全部由停用词组成的句子
        for i in context:
            words=jieba.cut(i)      #jieba进行分词
            words_filter=[word for word in words if word[0] not in stopkey]     #停用词的过滤
            if len(words_filter) != 0:
                total_cutword.append(words_filter)      #加入列表
                total_content.append(i)     #对非全停用词的句子加入
        return total_cutword,total_content

    #过滤单词
    def filter_model(sents,model):
        # 过滤词汇表中没有的单词
        total = []
        for sentence_i in sents:
            sentence_list = []
            for word_j in sentence_i:
                if word_j in model and len(word_j)>1: #过滤长度小于2的词
                    sentence_list.append(word_j)
            total.append(sentence_list)
        return total

    def cosine_similarity(vec1,vec2):
        # 计算两个向量之间的余弦相似度
        '''
        :param vec1: 向量1
        :param vec2: 向量2
        :return: 余弦相似度
        '''
        tx =np.array(vec1)
        ty = np.array(vec2)
        cos1 = np.sum(tx * ty)
        cos21 = np.sqrt(sum(tx ** 2))
        cos22 = np.sqrt(sum(ty ** 2))
        cosine_value = cos1/float(cos21 * cos22)
        return cosine_value


    def computer_similarity_by_avg(sents_1,sents_2,model):
        '''
        对两个句子求平均词向量作为边权重
        '''
        if len(sents_1) ==0 or len(sents_2) == 0:
            return 0.0
        vec1_avg = sum(model[word] for word in sents_1) / len(sents_1)
        vec2_avg = sum(model[word] for word in sents_2) / len(sents_2)

        similarity = tr.cosine_similarity(vec1_avg , vec2_avg)
        return similarity


    def create_graph(word_sent,model):
        '''
        传入句子链表，返回句子之间相似度的图
        '''
        num = len(word_sent)        #num表示句子数目
        board = np.zeros((num,num))     #以句子为节点构建句子间的相似矩阵

        #用余弦相似度初始化相似度矩阵
        for i,j in itertools.product(range(num), repeat=2):        #product返回A、B中的元素的笛卡尔积的元组，有向图
            if i != j:  #处理每个节点与其他节点的关联
                board[i][j] = tr.computer_similarity_by_avg(word_sent[i], word_sent[j],model)      #相似矩阵

        for i in word_sent:     #对含有关键词语的句子增加权重
            if "实用新型" in i:
                board[word_sent.index(i)][word_sent.index(i)]+=50.0
        board[0][0] += 70.0  # 对每段的第一句增加权重
        return board


    def sorted_sentence(graph,sentences,topK):
        '''
        调用pagerank算法进行计算，并排序
        '''
        key_index = []
        key_sentences = []

        nx_graph = nx.from_numpy_matrix(graph)
        #pagerank_numpy默认的阻尼d为0.85
        scores = nx.pagerank_numpy(nx_graph)

        sorted_scores = sorted(scores.items(), key = lambda item:item[1],reverse=True)
        for index,_ in sorted_scores[:topK]:
            key_index.append(index)
        new_index = sorted(key_index)       #对选出来的关键句索引进行顺序排序
        for i in new_index:
            key_sentences.append(sentences[i])
        return key_sentences

    #读取数据并分段
    def LoadData():
        with open('data/content.txt',encoding='GBK') as f:
            content = "".join(line.strip()+"\n" for line in f.readlines() if line)
        paragraphs = content.split("\n")
        if "" in paragraphs:
            paragraphs.remove("")
        return paragraphs

    #textrank算法
    def Run(text,topK):
        '''
        :param text: 输入文本
        :param topK: 选用前topK个句子作为摘要
        '''
        list_sents = tr.cut_sents(text)        #list_sents是输入的句子的列表
        data,sentences = tr.cut_word(list_sents)       #其中data为每个句子的词语列表的列表，sentences是该段话的句子列表
        # 训练模型
        model = gensim.models.Word2Vec(data, size=256, window=5, iter=10, min_count=1, workers=4)
        #data是训练参数,min_count是需要计算词向量的最小词频,size是隐藏神经元的个数,window词向量上下文最大距离
        sents2 = tr.filter_model(data,model)
        graph = tr.create_graph(sents2,model)      #graph是句子的相似矩阵
        result_sentence = tr.sorted_sentence(graph,sentences,topK)
        summsry = "，".join(result_sentence)
        context = {}
        context['summsry'] = summsry
        #关键词提取
        key_list = sentences[0].split("一种")
        if len(key_list)>1:
            key_sentence = "".join(key_list[len(key_list)-1])
        else:
            key_sentence = "".join(key_list)
        key_words = key_sentence.split("的")
        if len(key_words)>1:
            key_word = "".join(key_words[len(key_words)-1])
        else:
            key_word = "".join(key_words)
        context['keyword'] = key_word
        return context

def basic(request):
    content={}
    return render(request,'t_rank.html',content)

def res(request):
    content = {}
    content['org'] = ""
    if request.POST:
        tmp = request.POST['q']
        content['org'] = tmp
        if tmp != "":
            content['result'] = tr.Run(tmp,2)
    return render(request,'search.html',content)