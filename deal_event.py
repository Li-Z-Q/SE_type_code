import ast
import json


if __name__ == '__main__':
    old_file = open('event_annotate/events.txt', 'r', encoding='utf-8')
    annotate_file = open('event_annotate/events_annotate.txt', 'w', encoding='utf-8')
    uncertain_file = open('event_annotate/events_uncertain.txt', 'w', encoding='utf-8')

    i = 0
    for line in old_file:

        data_dict = ast.literal_eval(line)
        print(data_dict['eventName'])
        print(data_dict['desc'])

        annotation = input('i={0}, annotation= '.format(i))

        data_dict['label'] = annotation
        data_str = json.dumps(data_dict, ensure_ascii=False)

        if annotation == '2':
            uncertain_file.write(data_str + '\n')
            # uncertain_file.write('\r\n')
        else:
            annotate_file.write(data_str + '\n')
            # annotate_file.write('\r\n')

        i += 1
        if i >= 150:
            break

        annotate_file.flush()
        uncertain_file.flush()

        print()

    annotate_file.close()
    uncertain_file.close()
