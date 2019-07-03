import unidecode
x = []
DATA_HOME = "C:/Users/592123/Documents/DS Projects/EmailGenLSTM/data/"
with open(DATA_HOME + 'data.txt', 'r') as f:
    x = f.readlines()
print("hi")

#%%
y = []
for i in x:
    if i != '\n':
        y.append(i)
#%%
filters = ('Return-Path',
           'X-Sieve',
           'Message-Id',
           'From',
           'Reply-To',
           'To',
           'Date',
           'Subject',
           'X-Mailer',
           'MIME-Version',
           'Content-Type',
           'Content-Transfer-Encoding',
           'X-MIME-Autoconverted',
           'Status')
filtered = list(filter(lambda q: q.startswith(filters) == False, y))
#%%
filtered_str = ''.join(filtered)
#%%
unaccented_string = unidecode.unidecode(text).lower()
#%%
words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
print(len(words))
#%%
vocab = sorted(set(whatisthat))
print(len(vocab))
#%%
whatisthat = re.sub("\t"," ",whatisthat)