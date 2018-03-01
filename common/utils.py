import re

######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

replacements = []

with open('utils/nlp/mapping.pair') as f:
    for line in f.readlines():
        tok_from, tok_to = line.replace('\n','').split('\t')
        replacements.append((' '+tok_from+' ',' '+tok_to+' '))

def insertSpace(token,text):
    sidx = 0
    while True:
        sidx = text.find(token,sidx)
        if sidx==-1:
            break
        if sidx+1<len(text) and re.match('[0-9]',text[sidx-1]) and \
                re.match('[0-9]',text[sidx+1]):
            sidx += 1
            continue
        if text[sidx-1]!=' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx +=1
        if sidx+len(token)<len(text) and text[sidx+len(token)]!=' ':
            text = text[:sidx+1] + ' ' + text[sidx+1:]
        sidx+=1
    return text

def normalize(text):

    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$','',text)
    
    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0],sidx)
            if text[sidx-1]=='(':
                sidx -= 1
            eidx = text.find(m[-1],sidx)+len(m[-1])
            text = text.replace(text[sidx:eidx],''.join(m))
    
    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m,sidx)
            eidx = sidx + len(m)
            text = text[:sidx]+re.sub('[,\. ]','',m)+text[eidx:]

    # replace st.
    text = text.replace(';',',')
    text = re.sub('$\/','',text)
    text = text.replace('/',' and ')

    # replace other special characters
    text = text.replace('-',' ')
    text = re.sub('[\":\<>@\(\)]','',text)

    # insert white space before and after tokens:
    for token in ['?','.',',','!']:
        text = insertSpace(token,text)
    
    # insert white space for 's
    text = insertSpace('\'s',text)
     
    # replace it's, does't, you'd ... etc
    text = re.sub('^\'','',text)
    text = re.sub('\'$','',text)
    text = re.sub('\'\s',' ',text)
    text = re.sub('\s\'',' ',text)
    for fromx, tox in replacements:
		text = ' '+text+' '
		text = text.replace(fromx,tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +',' ',text)
    
    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i<len(tokens):
        if  re.match(u'^\d+$',tokens[i]) and \
            re.match(u'\d+$',tokens[i-1]):
            tokens[i-1]+=tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)
    
    return text
