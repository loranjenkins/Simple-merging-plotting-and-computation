from pathlib import Path

from natsort import natsorted

if __name__ == '__main__':
    condition_60_40 = [0, 10, 13, 23, 27, 37, 41, 51]
    condition_40_60 = [1, 6, 14, 20, 28, 34, 42, 48]
    condition_55_45 = [4, 8, 17, 22, 32, 36, 45, 50]
    condition_45_55 = [3, 11, 16, 24, 31, 39, 44, 52]

    condition_50_50 = [2, 5, 7, 12, 15, 19, 21, 25, 30, 33, 35, 40, 43, 46, 49, 53]

    files_directory1 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 1 - time_12-13 -10-10-2022\Final_data'
    files_directory2 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 2 - time_1400-1500 - 10-10-2022\Final_data'
    files_directory3 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 3 - time_ 10_30 - 18-10-2022\Final_data'
    files_directory4 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 4 - time_ 12_00 - 18-10-2022\Final_data'
    files_directory5 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 5 - time 10_30 - 19-10-2022\Final_data'
    files_directory6 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 6 - time 14 - 21-10-2022\Final_data'
    files_directory7 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Experiment 7 - time 1400 - 26-10-2022\Final_data'


    trails1 = []
    trails2 = []
    trails3 = []
    trails4 = []
    trails5 = []
    trails6 = []
    trails7 = []

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

    for file in Path(files_directory7).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails7.append(file)

    trails1 = natsorted(trails1, key=str)
    trails2 = natsorted(trails2, key=str)
    trails3 = natsorted(trails3, key=str)
    trails4 = natsorted(trails4, key=str)
    trails5 = natsorted(trails5, key=str)
    trails6 = natsorted(trails6, key=str)
    trails7 = natsorted(trails7, key=str)
    # trails8 = natsorted(trails8, key=str)
    print(len(trails2))
    all_conditions_60_40 = []
    for i in condition_60_40:
        all_conditions_60_40.append(trails1[i])
        all_conditions_60_40.append(trails2[i])
        all_conditions_60_40.append(trails3[i])
        all_conditions_60_40.append(trails4[i])
        all_conditions_60_40.append(trails5[i])
        all_conditions_60_40.append(trails6[i])
        all_conditions_60_40.append(trails7[i])
        # all_conditions_60_40.append(trails8[i])

    all_conditions_40_60 = []
    for i in condition_60_40:
        all_conditions_40_60.append(trails1[i])
        all_conditions_40_60.append(trails2[i])
        all_conditions_40_60.append(trails3[i])
        all_conditions_40_60.append(trails4[i])
        all_conditions_40_60.append(trails5[i])
        all_conditions_40_60.append(trails6[i])
        all_conditions_40_60.append(trails7[i])
        # all_conditions_50_50.append(trails8[i])

    all_conditions_55_45 = []
    for i in condition_55_45:
        all_conditions_55_45.append(trails1[i])
        all_conditions_55_45.append(trails2[i])
        all_conditions_55_45.append(trails3[i])
        all_conditions_55_45.append(trails4[i])
        all_conditions_55_45.append(trails5[i])
        all_conditions_55_45.append(trails6[i])
        all_conditions_55_45.append(trails7[i])

    all_conditions_45_55 = []
    for i in condition_45_55:
        all_conditions_45_55.append(trails1[i])
        all_conditions_45_55.append(trails2[i])
        all_conditions_45_55.append(trails3[i])
        all_conditions_45_55.append(trails4[i])
        all_conditions_45_55.append(trails5[i])
        all_conditions_45_55.append(trails6[i])
        all_conditions_45_55.append(trails7[i])

    all_conditions_50_50 = []
    for i in condition_50_50:
        all_conditions_50_50.append(trails1[i])
        all_conditions_50_50.append(trails2[i])
        all_conditions_50_50.append(trails3[i])
        all_conditions_50_50.append(trails4[i])
        all_conditions_50_50.append(trails5[i])
        all_conditions_50_50.append(trails6[i])
        all_conditions_50_50.append(trails7[i])


    root1_right = Path(r"C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\right")
    root2_left = Path(r"C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\left")

    root3_right = Path(r"C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\right")
    root4_left = Path(r"C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\left")

    root5 = Path(r"C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50")


    for file in all_conditions_60_40:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root1_right / file.name)

    for file in all_conditions_40_60:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root2_left / file.name)

    for file in all_conditions_55_55:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root3_right / file.name)

    for file in all_conditions_45_55:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root4_left / file.name)

    for file in all_conditions_50_50:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root5 / file.name)