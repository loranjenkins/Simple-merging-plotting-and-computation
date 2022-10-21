from pathlib import Path

if __name__ == '__main__':
    condition_60_40 = [0, 1, 6, 10, 13, 14, 20, 23, 27, 28, 34, 37, 41, 42, 48, 51]
    condition_55_45 = [3, 4, 8, 11, 16, 17, 22, 24, 31, 32, 36, 39, 44, 45, 50, 53]
    condition_50_50 = [2, 5, 7, 12, 15, 19, 21, 25, 30, 33, 35, 40, 43, 46, 49, 52]

    files_directory1 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder1'
    files_directory2 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder2'
    # files_directory3 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder3'
    # files_directory4 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder4'
    # files_directory5 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder5'
    # files_directory6 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder6'
    # files_directory7 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder7'
    # files_directory8 = r'C:\Users\loran\Desktop\data_formatter\JOAN_folder8'

    trails1 = []
    trails2 = []
    # trails3 = []
    # trails4 = []
    # trails5 = []
    # trails6 = []
    # trails7 = []
    # trails8 = []

    for file in Path(files_directory1).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails1.append(file)

    for file in Path(files_directory2).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails2.append(file)

    # for file in Path(files_directory3).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails3.append(file)
    #
    # for file in Path(files_directory4).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails4.append(file)
    #
    # for file in Path(files_directory5).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails5.append(file)
    #
    # for file in Path(files_directory6).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails6.append(file)
    #
    # for file in Path(files_directory7).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails7.append(file)
    #
    # for file in Path(files_directory8).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails8.append(file)


    all_conditions_60_40 = []
    for i in condition_60_40:
        all_conditions_60_40.append(trails1[i])
        all_conditions_60_40.append(trails2[i])
        # all_conditions_60_40.append(trails3[i])
        # all_conditions_60_40.append(trails4[i])
        # all_conditions_60_40.append(trails5[i])
        # all_conditions_60_40.append(trails6[i])
        # all_conditions_60_40.append(trails7[i])
        # all_conditions_60_40.append(trails8[i])

    all_conditions_50_50 = []
    for i in condition_50_50:
        all_conditions_50_50.append(trails1[i])
        all_conditions_50_50.append(trails2[i])
        # all_conditions_50_50.append(trails3[i])
        # all_conditions_50_50.append(trails4[i])
        # all_conditions_50_50.append(trails5[i])
        # all_conditions_50_50.append(trails6[i])
        # all_conditions_50_50.append(trails7[i])
        # all_conditions_50_50.append(trails8[i])

    all_conditions_55_45 = []
    for i in condition_55_45:
        all_conditions_55_45.append(trails1[i])
        all_conditions_55_45.append(trails2[i])
        # all_conditions_55_45.append(trails3[i])
        # all_conditions_55_45.append(trails4[i])
        # all_conditions_55_45.append(trails5[i])
        # all_conditions_55_45.append(trails6[i])
        # all_conditions_55_45.append(trails7[i])
        # all_conditions_55_45.append(trails8[i])


    root1 = Path("C:\\Users\loran\Desktop\data_formatter\condition_60_40")
    root2 = Path("C:\\Users\loran\Desktop\data_formatter\condition_50_50")
    root3 = Path("C:\\Users\loran\Desktop\data_formatter\condition_55_45")

    for file in all_conditions_60_40:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root1 / file.name)

    for file in all_conditions_50_50:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root2 / file.name)

    for file in all_conditions_55_45:
        folder_name = file.stem.rsplit("_", 1)[-1]
        file.rename(root3 / file.name)