import torch
from torch import nn
from torchtext.datasets import AG_NEWS
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

def zad():
    text = "I'm having a wonderful time at WZUM laboratories!"
    # print(text.split())
    tokenizer = get_tokenizer("basic_english")
    vocabulary = {word: i for i, word in enumerate(set(tokenizer(text)))}
    print(vocabulary) # ['time'])
    print(tokenizer(text))

    t1 = "John likes to watch movies. Mary likes movies too."
    t2 = "Mary also likes to watch football games."
    t1 = tokenizer(t1)
    t2 = tokenizer(t2)
    vocab = set(t1 + t2)
    vocab = dict.fromkeys(vocab, 0)
    # print(vocab)

    t1_vocab = vocab.copy()
    for word in t1:
        t1_vocab[word] += 1

    t2_vocab = vocab.copy()
    for word in t2:
        t2_vocab[word] += 1

    # print(t1_vocab)
    # print(t2_vocab)

    t1 = list(t1_vocab.values())
    t2 = list(t2_vocab.values())
    # print(t1)

def BoW():
    tokenizer = get_tokenizer("basic_english")

    t1 = "John likes to watch movies. Mary likes movies too."
    t2 = "Mary also likes to watch football games."

    def yield_tokens(data):  # generator zwracający tokenizowany tekst
        for t in data:
            yield tokenizer(t)

    vocab = build_vocab_from_iterator(yield_tokens([t1, t2]), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])  # ustawienie, że dla nieznanych słów słownik ma zwracać indeks 0

    print(vocab["football"])
    print(vocab["himalaje"])

def zadania():
# ZADANIE 1 Zainstaluj bibliotekę torchtext i zaimportuj z niej dataset AG_NEWS.
    # Dane są już podzielone na zbiór testowy i treningowy i przechowywane jako tuple iteratorów.
    # train_data, test_data = AG_NEWS()
    # print(next(train_data)[1])
    dataset = AG_NEWS(split='train')
    # print(dataset)
    # print(type(dataset))
    # print(next(dataset))

    # for data in dataset:
    #    print(data)

# ZADANIE 2 utwórz tokenizer typu basic_english
# utwórz słownik typu Vocab na podstawie tekstów zawartych w zbiorze treningowym
# stwórz pipeline do preprocessingu tekstu (text_pipeline) przy użyciu funkcji lambda.
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(iter):  # generator zwracający tokenizowany tekst
        for _, text in iter:
            yield tokenizer(text)

    # print(next(yield_tokens(dataset)))

    vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])  # ustawienie, że dla nieznanych słów słownik ma zwracać indeks 0
    tekst = "I'm having a wonderful time at WZUM laboratories!"
    # print(vocab(tokenizer(tekst)))

    text_pipeline = lambda x: vocab(tokenizer(x))
    # print(text_pipeline(tekst))

# ZADANIE 4 Utwórz DataLoader wykorzystujący zbiór treningowy.
    dataset = AG_NEWS(split='train')

    def collate_batch(batch):
        # print(len(batch))
        # exit()
        label_list, text_list, offset = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(_label-1)
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offset.append(processed_text.size(0))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.cat(text_list)
        offset = torch.tensor(offset[:-1]).cumsum(dim=0)
        # print(label_list.shape, text_list.shape, offset.shape)
        return label_list, text_list, offset

    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_batch) # True

    # for data in data_loader:
        # print(data)

# ZADANIE 5 Utwórz model składający się z jednej warstwy EmbeddingBag i jednej warstwy Linear.
    class KlasyfikatorTekstu3000(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            num_class = 4
            super(KlasyfikatorTekstu3000, self).__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
            self.fc = nn.Linear(embed_dim, num_class)
            self.loss_function = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.parameters(), lr=5)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, 0.1)

        def forward(self, text, offsets):
            out = self.embedding(text, offsets)
            return self.fc(out)

    model = KlasyfikatorTekstu3000(len(vocab), 5)
    # for labels, texts, offsets in data_loader:
    #     print(model(texts, offsets))
    #     exit()

# ZADANIE 6 Utwórz funkcję służącą do treningu modelu.
# Utwórz funkcję służącą do ewaluacji modelu.
    def train(data_loader):
        model.train()
        total_acc, total_count = 0, 0
        for idx, (label, text, offset) in enumerate(data_loader):
            model.optimizer.zero_grad()
            pred_label = model(text, offset)
            loss = model.loss_function(pred_label, label)
            loss.backward()
            model.optimizer.step()

            total_acc += (pred_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

            if idx % 500 == 0:
                print(f'accuracy for {idx} in {len(data_loader)} : {total_acc/total_count*100}')
                total_acc, total_count = 0, 0

    def evaluate(data_loader):
        model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for idx, (label, text, offset) in enumerate(data_loader):
                pred_label = model(text, offset)
                total_acc += (pred_label.argmax(1) == label).sum().item()
                total_count += label.size(0)

        return total_acc/total_count

# ZADANIE 7 zdefiniuj funkcję strat jako CrossEntropyLoss, a optimizer jako SGD, dobierz learning_rate.
# używając funkcji to_map_style_dataset zmień typ zapisu datasetu z iteratora na zwykły wektor.
# utwórz nowy DataLoader treningowy i testowy wykorzystujący zmapowany dataset. Parametr shuffle ustaw na True.
    BATCH_SIZE = 64
    EPOCHS = 10
    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# ZADANIE 8 Przeprowadź uczenie dla ok 10 epok, wyświetlaj skuteczność co epokę.
    for epoch in range(1, EPOCHS+1):
        train(train_dataloader)
        accu_val = evaluate(test_dataloader)
        model.scheduler.step()
        print("-"*20)
        print(f'Accuracy for epoch {epoch} : {accu_val*100}')
        print("-" * 20)

# ZADANIE 9 Sprawdź działanie modelu dla newsu znalezionego w sieci.
    ag_news_label = {1: 'World',
                     2: 'Sport',
                     3: 'Business',
                     4: 'Sci/Tec'}

    def predict(text):
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() +1

    my_fav_news =  'The postponement of Manchester United’s match at Brentford affects two of the most expensive, and popular, players in Fantasy Premier League.'

    model = model.to('cpu')
    print(f'This news is about : {ag_news_label[predict(my_fav_news)]}')


def main():
    # zad()
    # BoW()
    zadania()

if __name__ == '__main__':
    main()