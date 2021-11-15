import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def creating_empty_table():
    tuples = list(
        zip(
            *[
                ["epoch0", "epoch0", "epoch1", "epoch1", "epoch2", "epoch2", "epoch3", "epoch3", "epoch4", "epoch4", "epoch5", "epoch5", "epoch6", "epoch6", "epoch7", "epoch7", "epoch8", "epoch8", "epoch9", "epoch9", "epoch10", "epoch10", "epoch11", "epoch11", "epoch12", "epoch12", "epoch13", "epoch13", "epoch14", "epoch14", "epoch15", "epoch15", "epoch16", "epoch16", "epoch17", "epoch17", "epoch18", "epoch18", "epoch19", "epoch19"],
                ["bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm", "bbox", "segm"],
            ]
        )
    )
    what_kind_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']


    index = pd.MultiIndex.from_tuples(tuples, names=["", ""])

    df_inmethod = pd.DataFrame(np.zeros((40, 12)), index=index, columns=what_kind_list)

    #df2 = df[:4]

    return df_inmethod

def change_table_value(df, counter, value_to_fill_in_table, epoch, bbox):

                    epoch_name = 'epoch'+str(epoch)
                    if bbox:
                        index_bbox_or_seg = 'bbox'
                    else:
                        index_bbox_or_seg = 'segm'

                    if counter == 0:
                        row = 'A'
                    elif counter == 1:
                        row = 'B'
                    elif counter == 2:
                        row = 'C'
                    elif counter == 3:
                        row = 'D'
                    elif counter == 4:
                        row = 'E'
                    elif counter == 5:
                        row = 'F'
                    elif counter == 6:
                        row = 'G'
                    elif counter == 7:
                        row = 'H'
                    elif counter == 8:
                        row = 'I'
                    elif counter == 9:
                        row = 'J'
                    elif counter == 10:
                        row = 'K'
                    else:
                        row = 'L'

                    #print(type(index_bbox_or_seg))

                    df[row].loc[epoch_name, index_bbox_or_seg] = value_to_fill_in_table
                    #df['A'].loc['[21-23)', 'M', '[10000-20000)']
                    #print(value_to_fill_in_table)
                    #print(df[row][epoch_name][index_bbox_or_seg])
                    #print(type(df[row][epoch_name][index_bbox_or_seg]))
                    #print(type(value_to_fill_in_table))
                    #print(row)
                    #print(epoch_name)
                    #print(index_bbox_or_seg)


def creating_table(df):
    final_epoch = 31
    epoch = 0
    counter = 0
    loop_in_bbox = True
    bbox = False
    counter_on = False

    with open("statsfile.txt", "r") as a_file:
        for line in a_file:
            stripped_line = line.strip()
            #print(stripped_line)
            if stripped_line.startswith('IoU metric: segm'):
                epoch += 1
                if epoch == final_epoch:
                    break
                loop_in_bbox = True
                counter = 0
                bbox = False
                counter_on = False
            if stripped_line.startswith('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]') and loop_in_bbox == True:
                bbox=True
                counter_on = True
                loop_in_bbox = False
            if stripped_line.startswith('IoU metric: bbox'):
                bbox=False
                counter = -1
            if counter_on == True:
                if counter == -1:
                    counter = 0
                else:
                    print(epoch)
                    change_table_value(df, counter, stripped_line[-6:], epoch, bbox)
                    counter += 1

def creating_bbox_segm_plots(letter, bbox_values, segm_values):

    #print(bbox_values)
    #print(segm_values)
    #print('')

    plt.clf()
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    y = bbox_values
    plt.plot(x, y, 'o')
    plt.title('bbox')
    plt.xlabel('epoch')
    plt.ylabel(letter)
    plt.xticks(x)
    plt.ylim(0,1)
    save_name = 'evaluation_folder/Bbox_figures/' + letter
    plt.savefig(save_name)
    plt.clf()
    y = segm_values
    plt.plot(x, y, 'o')
    plt.title('segm')
    plt.xlabel('epoch')
    plt.ylabel(letter)
    plt.xticks(x)
    plt.ylim(0, 1)
    save_name = 'evaluation_folder/Segm_figures/' + letter
    plt.savefig(save_name)
    '''
    print(bbox_values)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title("Setting range of Axes", fontsize=25)
    plt.xlabel("x", fontsize=18)
    plt.ylabel("1+sinx", fontsize=18)
    plt.xlim(4, 8)
    plt.ylim(-0.5, 2.5)
    plt.show()
    '''


def creating_plots(df):

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    epochs = ['epoch0', 'epoch1', 'epoch2', 'epoch3', 'epoch4', 'epoch5', 'epoch6', 'epoch7', 'epoch8', 'epoch9', 'epoch10', 'epoch11', 'epoch12', 'epoch13', 'epoch14', 'epoch15', 'epoch16', 'epoch17', 'epoch18', 'epoch19']

    for letter in letters:
        bbox_values = []
        segm_values = []
        for epoch in epochs:
            bbox_values.append(float(df[letter][epoch]['bbox']))
            segm_values.append(float(df[letter][epoch]['segm']))
        creating_bbox_segm_plots(letter, bbox_values, segm_values)


df = creating_empty_table()
creating_table(df)
df.to_csv('evaluation_folder/table_bbox_segm.csv')
creating_plots(df)
