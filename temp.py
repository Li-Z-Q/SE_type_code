raw_file = open('data/weibo_events_annotate.txt', 'r', encoding='utf-8')

for line in raw_file:

    raw_data = line.strip('\n').split(', ')
    print(raw_data)
