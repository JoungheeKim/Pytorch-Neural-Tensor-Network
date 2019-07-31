import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence

class DocumentNTN(nn.Module):
    def __init__(self, dictionary_size, embedding_size, tensor_dim,
                 num_class, hidden_size, attention_size, n_layers=1, dropout_p=0.05, device="cpu"):
        super(DocumentNTN, self).__init__()

        self.device = device
        self.ntn = NeuralTensorNetwork(dictionary_size=dictionary_size,
                                       embedding_size=embedding_size,
                                       tensor_dim=tensor_dim,
                                       dropout=dropout_p,
                                       device=device)

        self.rnn = nn.GRU(input_size=tensor_dim,
                          hidden_size=int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=True,
                          batch_first=True
                          )

        self.attn = Attention(hidden_size=hidden_size,
                               attention_size=attention_size)

        self.output = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def set_embedding(self, embedding, requires_grad = True):
        self.ntn.emb.weight.data.copy_(embedding)
        return True

    def forward(self, document, sentence_per_document, svo_length_per_sentence):
        batch_size, max_sentence_length, max_word_length = document.size()
        # |document| = (batch_size, max_sentence_length, max_word_length)
        # |sentence_per_document| = (batch_size)
        # |word_per_sentence| = (batch_size, max_sentence_length)
        # |svo_length_per_sentence| = (batch_size, max_sentence_length, 3)

        #print("제발", sentence_per_document)

        # Remove sentence-padding in document by using "pack_padded_sequence.data"
        packed_sentences = pack(document,
                                lengths=sentence_per_document.tolist(),
                                batch_first=True,
                                enforce_sorted=False)
        # |packed_sentences.data| = (sum(sentence_length), max_word_length)

        # Remove sentence-padding in svo_length_per_sentence "pack_padded_sequence.data"
        packed_svo_length_per_sentence = pack(svo_length_per_sentence,
                                         lengths=sentence_per_document.tolist(),
                                         batch_first=True,
                                         enforce_sorted=False)
        # |packed_svo_length_per_sentence.data| = (sum(sentence_length), 3)

        sentence_vecs = self.ntn(packed_sentences.data, packed_svo_length_per_sentence.data)
        # |sentence_vecs| = (sum(sentence_length), tensor_dim)

        # "packed_sentences" have same information to recover PackedSequence for sentence
        packed_sentence_vecs = PackedSequence(data=sentence_vecs,
                                              batch_sizes=packed_sentences.batch_sizes,
                                              sorted_indices=packed_sentences.sorted_indices,
                                              unsorted_indices=packed_sentences.unsorted_indices)
        # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
        mask = self.generate_mask(sentence_per_document)
        # |mask| = (batch_size, max(sentence_per_document))

        # Get document vectors By using GRU
        last_hiddens, _ = self.rnn(packed_sentence_vecs)

        # Unpack ouput of rnn model
        last_hiddens, _ = unpack(last_hiddens, batch_first=True)
        # |last_hiddens| = (batch_size, max(sentence_per_document), hidden_size)

        # Get attention weights and context vectors
        context_vectors, context_weights = self.attn(last_hiddens, mask)
        # |context_vectors| = (batch_size, hidden_size)
        # |context_weights| = (batch_size, max(sentence_per_document))

        y = self.softmax(self.output(context_vectors))

        return y, context_weights

    def generate_mask(self, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples,
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat(
                    [torch.zeros((1, l), dtype=torch.uint8), torch.ones((1, (max_length - l)), dtype=torch.uint8)],
                    dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [torch.zeros((1, l), dtype=torch.uint8)]

        mask = torch.cat(mask, dim=0).byte()

        return mask.to(self.device)

class NeuralTensorNetwork(nn.Module):

    def __init__(self, dictionary_size, embedding_size, tensor_dim, dropout, device="cpu"):
        super(NeuralTensorNetwork, self).__init__()

        self.device = device
        self.emb = nn.Embedding(dictionary_size, embedding_size)
        self.tensor_dim = tensor_dim

        ##Tensor Weight
        # |T1| = (embedding_size, embedding_size, tensor_dim)
        self.T1 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T1.data.normal_(mean=0.0, std=0.02)

        # |T2| = (embedding_size, embedding_size, tensor_dim)
        self.T2 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T2.data.normal_(mean=0.0, std=0.02)

        # |T3| = (tensor_dim, tensor_dim, tensor_dim)
        self.T3 = nn.Parameter(torch.Tensor(tensor_dim * tensor_dim * tensor_dim))
        self.T3.data.normal_(mean=0.0, std=0.02)

        # |W1| = (embedding_size * 2, tensor_dim)
        self.W1 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W2| = (embedding_size * 2, tensor_dim)
        self.W2 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W3| = (tensor_dim * 2, tensor_dim)
        self.W3 = nn.Linear(tensor_dim * 2, tensor_dim)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, svo, sov_length):
        # |svo| = (batch_size, max_length)
        # |sov_length| = (batch_size, 3)

        svo = self.emb(svo)
        # |svo| = (batch_size, max_lenght, embedding_size)

        ## To merge word embeddings, Get mean value
        subj, verb, obj = [], [], []
        for batch_index, svo_batch in enumerate(sov_length):
            sub_svo = svo[batch_index]
            len_s, len_v, len_o = svo_batch
            subj += [torch.mean(sub_svo[:len_s], dim=0, keepdim=True)]
            verb += [torch.mean(sub_svo[len_s:len_s+len_v], dim=0, keepdim=True)]
            obj += [torch.mean(sub_svo[len_s+len_v:len_s+len_v+len_o], dim=0, keepdim=True)]

        subj = torch.cat(subj, dim=0)
        verb = torch.cat(verb, dim=0)
        obj = torch.cat(obj, dim=0)
        # |subj|, |verb|, |obj| = (batch_size, embedding_size)

        R1 = self.tensor_Linear(subj, verb, self.T1, self.W1)
        R1 = self.tanh(R1)
        R1 = self.dropout(R1)
        # |R1| = (batch_size, tensor_dim)

        R2 = self.tensor_Linear(verb, obj, self.T2, self.W2)
        R2 = self.tanh(R2)
        R2 = self.dropout(R2)
        # |R2| = (batch_size, tensor_dim)

        U = self.tensor_Linear(R1, R2, self.T3, self.W3)
        U = self.tanh(U)

        return U


    def tensor_Linear(self, o1, o2, tensor_layer, linear_layer):
        # |o1| = (batch_size, unknown_dim)
        # |o2| = (batch_size, unknown_dim)
        # |tensor_layer| = (unknown_dim * unknown_dim * tensor_dim)
        # |linear_layer| = (unknown_dim * 2, tensor_dim)

        batch_size, unknown_dim = o1.size()

        # 1. Linear Production
        o1_o2 = torch.cat((o1, o2), dim=1)
        # |o1_o2| = (batch_size, unknown_dim * 2)
        linear_product = linear_layer(o1_o2)
        # |linear_product| = (batch_size, tensor_dim)

        # 2. Tensor Production
        tensor_product = o1.mm(tensor_layer.view(unknown_dim, -1))
        # |tensor_product| = (batch_size, unknown_dim * tensor_dim)
        tensor_product = tensor_product.view(batch_size, -1, unknown_dim).bmm(o2.unsqueeze(1).permute(0,2,1).contiguous()).squeeze()
        tensor_product = tensor_product.contiguous()
        # |tensor_product| = (batch_size, tensor_dim)

        # 3. Summation
        result = tensor_product + linear_product
        # |result| = (batch_size, tensor_dim)

        return result

class Attention(nn.Module):

    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, attention_size, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        ## Context vector
        self.context_weight = nn.Parameter(torch.Tensor(attention_size, 1))
        self.context_weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, h_src, mask=None):
        # |h_src| = (batch_size, length, hidden_size)
        # |mask| = (batch_size, length)
        batch_size, length, hidden_size = h_src.size()

        # Resize hidden_vectors to generate weight
        weights = h_src.view(-1, hidden_size)
        weights = self.linear(weights)
        weights = self.tanh(weights)

        weights = torch.mm(weights, self.context_weight).view(batch_size, -1)
        # |weights| = (batch_size, length)

        if mask is not None:
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0, masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch, the weight for empty time-step would be set to 0.
            weights.masked_fill_(mask, -float('inf'))

        # Modified every values to (0~1) by using softmax function
        weights = self.softmax(weights)
        # |weights| = (batch_size, length)

        context_vectors = torch.bmm(weights.unsqueeze(1), h_src)
        # |context_vector| = (batch_size, 1, hidden_size)

        context_vectors = context_vectors.squeeze(1)
        # |context_vector| = (batch_size, hidden_size)

        return context_vectors, weights

