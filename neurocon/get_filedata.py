import os
def get_filepath(path):
    folder_path = path
    files = os.listdir(folder_path)
    control_anat_data = {}
    control_func_data = {}
    control_anat_num = 0
    control_func_num = 0
    patient_anat_data = {}
    patient_func_data = {}
    patient_anat_num = 0
    patient_func_num = 0

    for folder in os.listdir(folder_path):
        if not folder.startswith('.'): # 排除隐藏文件夹
            patient_folder = os.path.join(folder_path, folder)#
            if os.path.isdir(patient_folder):
                # 遍历每个患者文件夹下的anat和func文件夹
                for subfolder in os.listdir(patient_folder):
                    subfolder_path = os.path.join(patient_folder, subfolder)
                    if os.path.isdir(subfolder_path):
                        # 遍历anat和func文件夹下的所有文件
                        for file in os.listdir(subfolder_path):
                            file_path = os.path.join(subfolder_path, file)
                            if file.endswith('.nii.gz'):
                                # 如果是nii.gz格式的文件，则读取数据并存储到相应的变量中
                                if folder.startswith('sub-control'):
                                    if subfolder == 'anat':
                                        control_anat_data[control_anat_num] = file_path
                                        control_anat_num = control_anat_num + 1
                                    elif subfolder == 'func':
                                        control_func_data[control_func_num] = file_path
                                        control_func_num = control_func_num + 1
                                elif folder.startswith('sub-patient'):
                                    if subfolder == 'anat':
                                        patient_anat_data[patient_anat_num] = file_path
                                        patient_anat_num = patient_anat_num + 1
                                    elif subfolder == 'func':
                                        patient_func_data[patient_func_num] = file_path
                                        patient_func_num = patient_func_num + 1

    '''    print('patient_folder:',patient_folder)

    print('controlanat_num:',control_anat_num,'controlanat_data',control_anat_data)
    print('controlfunc_num:',control_func_num,'controlfunc:data',control_func_data)
    print('patientanat_num:',patient_anat_num,'patientanat_data',patient_anat_data)
    print('patientfunc_num:',patient_func_num,'patientfunc:data',patient_func_data)'''
    return control_anat_data,control_func_data,patient_anat_data,patient_func_data



