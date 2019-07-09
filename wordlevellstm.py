import re
DATA_HOME = "C:/Users/Daniel Atkinson/Desktop/DS/EmailGen/"

with open(DATA_HOME + 'clean_data.txt', 'r') as f:
    text = f.read().lower()
print('corpus length:', len(text))
#%%
text = re.sub("= ", " ", text)
print(len(text))
#%%
with open("red.txt", "w") as text_file:
    text_file.write(text)