import ast
import json


# def data_1():
#     old_file = open('event_annotate/events.txt', 'r', encoding='utf-8')
#     annotate_file = open('event_annotate/events_annotate.txt', 'w', encoding='utf-8')
#     uncertain_file = open('event_annotate/events_uncertain.txt', 'w', encoding='utf-8')
#
#     i = 0
#     for line in old_file:
#
#         data_dict = ast.literal_eval(line)
#         print(data_dict['eventName'])
#         print(data_dict['desc'])
#
#         annotation = input('i={0}, annotation= '.format(i))
#
#         data_dict['label'] = annotation
#         data_str = json.dumps(data_dict, ensure_ascii=False)
#
#         if annotation == '2':
#             uncertain_file.write(data_str + '\n')
#             # uncertain_file.write('\r\n')
#         else:
#             annotate_file.write(data_str + '\n')
#             # annotate_file.write('\r\n')
#
#         i += 1
#         if i >= 150:
#             break
#
#         annotate_file.flush()
#         uncertain_file.flush()
#
#         print()
#
#     annotate_file.close()
#     uncertain_file.close()

# def data_2():
#     old_file = open('event_annotate/1(1).txt', 'r', encoding='utf-8')
#     annotate_file = open('event_annotate/1(1)_annotate.txt', 'w', encoding='utf-8')
#     uncertain_file = open('event_annotate/1(1)_uncertain.txt', 'w', encoding='utf-8')
#
#     for line in old_file:
#
#         raw_data = line.strip('\n')
#         print(raw_data)
#
#         annotation = input('annotation= ')
#
#         data_annotate = raw_data + ', ' + str(annotation)
#
#         if annotation == '2':
#             uncertain_file.write(data_annotate + '\n')
#             # uncertain_file.write('\r\n')
#         else:
#             annotate_file.write(data_annotate + '\n')
#             # annotate_file.write('\r\n')
#
#         annotate_file.flush()
#         uncertain_file.flush()
#
#         print()
#
#     annotate_file.close()
#     uncertain_file.close()


def data_3():
    print('start read')
    old_file = open('event_annotate/weibo_events.txt', 'r', encoding='utf-8')
    annotate_file = open('event_annotate/weibo_events_annotate.txt', 'w', encoding='utf-8')
    uncertain_file = open('event_annotate/weibo_events_uncertain.txt', 'w', encoding='utf-8')

    i = 0
    for line in old_file:

        print('i: ', i)
        i += 1
        raw_data = line.strip('\n')
        print(raw_data)

        annotation = input('annotation= ')

        data_annotate = raw_data + ', ' + str(annotation)

        if annotation == '2':
            uncertain_file.write(data_annotate + '\n')
            # uncertain_file.write('\r\n')
        else:
            annotate_file.write(data_annotate + '\n')
            # annotate_file.write('\r\n')

        annotate_file.flush()
        uncertain_file.flush()

        print()

    annotate_file.close()
    uncertain_file.close()


if __name__ == '__main__':
    # with open("event_annotate/weibo_events.txt", "r", encoding='utf-8') as f:
    #     ftext = f.read()  # 一次性读全部成一个字符串
    #     print(ftext)
    data_3()