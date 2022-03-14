import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def creating_empty_table(number_of_epochs):
    '''This method creates and empty table with number of columns equal to number_of_epochs'''

    # Creating empty lists
    epoch_list = []
    bbox_seg_list = []

    # Filling list epoch_list with the name of every epoch twice ([epoch0, epoch0, epoch1, epoch1, epoch2,...]
    # Filling ist bbox_seg_list with 'bbox' and 'segm' for every epoch (['bbox', 'segm', 'epoch', 'segm', 'epoch',...]
    for i in range(number_of_epochs):
        epoch_name = 'epoch' + str(i)
        epoch_list.append(epoch_name)
        epoch_list.append(epoch_name)
        bbox_seg_list.append('bbox')
        bbox_seg_list.append('segm')

    # Creating a list with epoch_list and bbox_seg_list
    tuples = list(
        zip(
            *[
                epoch_list,
                bbox_seg_list
            ]
        )
    )

    index = pd.MultiIndex.from_tuples(tuples, names=["", ""])

    # Creating a list of letters, each letter representing a plot
    what_kind_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    # number of rows for plot
    number_of_rows = number_of_epochs * 2

    # Create the list with 12 rows and number_of_rows columns
    df_in_method = pd.DataFrame(np.zeros((number_of_rows, 12)), index=index, columns=what_kind_list)

    # Return the list
    return df_in_method


def change_table_value(df, counter, value_to_fill_in_table, epoch, bbox):
    '''Method to change values of table at specific position'''

    # Setting epoch_name to current epoch name
    epoch_name = 'epoch' + str(epoch)

    # Check if in bbox or segm, setting index_bbox_or_seg according to it
    if bbox:
        index_bbox_or_seg = 'bbox'
    else:
        index_bbox_or_seg = 'segm'

    # Depending to which counter at, setting to row name to specific letter
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

    # Fill table at the current position (row, epoch_name and index_bbox_or_seg)
    df[row].loc[epoch_name, index_bbox_or_seg] = value_to_fill_in_table


def creating_table(df, number_of_epochs):
    '''This methods writes values from statsfile.txt into table'''

    # For stopping table creation
    final_epoch = number_of_epochs + 1

    # Initializing epoch and counter at 0
    epoch = 0
    counter = 0

    # Setting booleans, which are important for creating the table in the right way
    loop_in_bbox = True
    bbox = False
    counter_on = False

    # Opening the statsfile.txt in mode 'reading'
    with open("statsfile.txt", "r") as a_file:

        # Looping through every line of statsfile.txt
        for line in a_file:
            # Calling method strip on every line
            stripped_line = line.strip()
            # If the line starts with 'IoU metric: segm'
            if stripped_line.startswith('IoU metric: segm'):
                # Increment epoch
                epoch += 1
                # If epoch equals final_epoch, creation of the table is done
                if epoch == final_epoch:
                    break
                # Setting loop_in_bbox to True, counter to zero and bbox to False
                loop_in_bbox = True
                counter = 0
                bbox = False
                counter_on = False
            # If the line starts with '...' and loop_in_bbox == True (now we plot for bbox)
            if stripped_line.startswith('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]') \
                    and loop_in_bbox:
                # Set bbox to True, bbox to True and loop_in_bbox to False
                bbox=True
                counter_on = True
                loop_in_bbox = False
            # If the line starts with 'IoU metric: bbox'
            if stripped_line.startswith('IoU metric: bbox'):
                # Set bbox to False, because now we are in the part for segm (segmentation)
                bbox=False
                # Decrement counter
                counter = -1
            # If counter is True
            if counter_on:
                # If counter equals -1, set counter to 0
                if counter == -1:
                    counter = 0
                # Else change the table value in the table df and increment counter
                else:
                    change_table_value(df, counter, stripped_line[-6:], epoch, bbox)
                    counter += 1

def creating_bbox_segm_plots(letter, bbox_values, segm_values, number_of_epochs):
    '''Method the create the bbox and segm plots'''

    # Clear the current figure
    plt.clf()

    # Create empty list for x-values
    x = []

    # For every epoch, append number to x
    for i in range(number_of_epochs):
        x.append(i)

    # Set y to the bbox_values
    y = bbox_values

    # plot x-y graph
    plt.plot(x, y, 'o')
    # Label plot
    plt.title('bbox')
    plt.xlabel('epoch')
    plt.ylabel(letter)
    plt.xticks(x)
    # Setting range for y-axis
    plt.ylim(0,1)
    # Saving figure
    save_name = 'evaluation_folder/Bbox_figures/' + letter
    plt.savefig(save_name)

    # Clear the current figure
    plt.clf()
    # Set y to the segm_values
    y = segm_values
    # plot x-y graph
    plt.plot(x, y, 'o')
    # Label plot
    plt.title('segm')
    plt.xlabel('epoch')
    plt.ylabel(letter)
    plt.xticks(x)
    # Setting range for y-axis
    plt.ylim(0, 1)
    # Saving figure
    save_name = 'evaluation_folder/Segm_figures/' + letter
    plt.savefig(save_name)


def epoch_table(number_of_epochs):
    '''Method to create a list of epochs depending on the number of epochs'''

    # Create empty list
    epochs = []

    # Fill list epochs with epoch names
    for i in range(number_of_epochs):
        epoch_name = 'epoch' + str(i)
        epochs.append(epoch_name)

    # Return the filled list
    return epochs


def creating_plots(df, number_of_epochs):
    '''Method to create plots from the table df'''

    # Creating table of letters for the table's rows
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    # Creating a list of the epochs for the table's columns
    epochs = epoch_table(number_of_epochs)

    # Creating two lists (for bbox and segm) and appending them with the values of the df at the letters, epochs and
    # bbox/segm. Finally, create the bbox and segm plots
    for letter in letters:
        bbox_values = []
        segm_values = []
        for epoch in epochs:
            bbox_values.append(float(df[letter][epoch]['bbox']))
            segm_values.append(float(df[letter][epoch]['segm']))
        creating_bbox_segm_plots(letter, bbox_values, segm_values, number_of_epochs)


if __name__ == '__main__':

    # Number of epochs for plotting
    number_of_epochs = 5

    # Creating an empty table with number_of_epochs columns
    df = creating_empty_table(number_of_epochs)

    # Filling values from statsfile.txt into table df
    creating_table(df, number_of_epochs)

    # Writing the table df into a csv file to the path 'evaluation_folder/table_bbox_segm.csv'
    df.to_csv('evaluation_folder/table_bbox_segm.csv')

    # Creating the plots from the table df
    creating_plots(df, number_of_epochs)
