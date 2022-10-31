from pathlib import Path

from natsort import natsorted

if __name__ == '__main__':
    condition_60_40 = [0, 1, 6, 10, 13, 14, 20, 23, 27, 28, 34, 37, 41, 42, 48, 51]
    condition_55_45 = [3, 4, 8, 11, 16, 17, 22, 24, 31, 32, 36, 39, 44, 45, 50, 52]
    condition_50_50 = [2, 5, 7, 12, 15, 19, 21, 25, 30, 33, 35, 40, 43, 46, 49, 53]

    files_directory1 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 1 - time_12-13 -10-10-2022\Final_data'
    files_directory2 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 2 - time_1400-1500 - 10-10-2022\Final_data'
    files_directory3 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 3 - time_ 10_30 - 18-10-2022\Final_data'
    files_directory4 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 4 - time_ 12_00 - 18-10-2022\Final_data'
    files_directory5 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 5 - time 10_30 - 19-10-2022\Final_data'
    files_directory6 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 6 - time 14 - 21-10-2022\Final_data'
    # files_directory7 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder7'
    # files_directory8 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder8'

    trails1 = []
    trails2 = []
    trails3 = []
    trails4 = []
    trails5 = []
    trails6 = []
    # trails7 = []
    # trails8 = []

    for file in Path(files_directory1).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails1.append(file)

    for file in Path(files_directory2).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails2.append(file)

    for file in Path(files_directory3).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails3.append(file)

    for file in Path(files_directory4).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails4.append(file)

    for file in Path(files_directory5).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails5.append(file)

    for file in Path(files_directory6).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails6.append(file)
    #
    # for file in Path(files_directory7).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails7.append(file)
    #
    # for file in Path(files_directory8).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails8.append(file)
    trails1 = natsorted(trails1, key=str)
    trails2 = natsorted(trails2, key=str)
    trails3 = natsorted(trails3, key=str)
    trails4 = natsorted(trails4, key=str)
    trails5 = natsorted(trails5, key=str)
    trails6 = natsorted(trails6, key=str)
    # trails7 = natsorted(trails7, key=str)
    # trails8 = natsorted(trails8, key=str)

    all_conditions_60_40 = []
    for i in condition_60_40:
        all_conditions_60_40.append(trails1[i])
        all_conditions_60_40.append(trails2[i])
        all_conditions_60_40.append(trails3[i])
        all_conditions_60_40.append(trails4[i])
        all_conditions_60_40.append(trails5[i])
        all_conditions_60_40.append(trails6[i])
        # all_conditions_60_40.append(trails7[i])
        # all_conditions_60_40.append(trails8[i])

    all_conditions_50_50 = []
    for i in condition_50_50:
        all_conditions_50_50.append(trails1[i])
        all_conditions_50_50.append(trails2[i])
        all_conditions_50_50.append(trails3[i])
        all_conditions_50_50.append(trails4[i])
        all_conditions_50_50.append(trails5[i])
        all_conditions_50_50.append(trails6[i])
        # all_conditions_50_50.append(trails7[i])
        # all_conditions_50_50.append(trails8[i])

    all_conditions_55_45 = []
    for i in condition_55_45:
        all_conditions_55_45.append(trails1[i])
        all_conditions_55_45.append(trails2[i])
        all_conditions_55_45.append(trails3[i])
        all_conditions_55_45.append(trails4[i])
        all_conditions_55_45.append(trails5[i])
        all_conditions_55_45.append(trails6[i])
        # all_conditions_55_45.append(trails7[i])
        # all_conditions_55_45.append(trails8[i])



    root1 = Path("C:\\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_60_40")
    root2 = Path("C:\\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_50_50")
    root3 = Path("C:\\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_55_45")

    for file in all_conditions_60_40:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root1 / file.name)

    for file in all_conditions_50_50:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root2 / file.name)

    for file in all_conditions_55_45:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root3 / file.name)