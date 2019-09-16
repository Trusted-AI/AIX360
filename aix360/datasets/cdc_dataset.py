import os
import sys
import requests
import pandas as pd
import xport, csv



def default_preprocessing(df):
    return df

class CDCDataset():
    """
    The CDC (Center for Disease Control and Prevention) questionnaire datasets [#]_ are surveys conducted
    by the organization involving 1000s of civilians about various facets of daily life. There are 44
    questionnaires that collect data about income, occupation, health, early childhood and many other
    behavioral and lifestyle aspects of people living in the US. These questionnaires are thus a rich
    source of information indicative of the quality of life of many civilians. More information about
    each questionaire and the type of answers are available in the following reference.

    References:
        .. [#] `NHANES 2013-2014 Questionnaire Data
           <https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&CycleBeginYear=2013>`_
    """

    def __init__(self, custom_preprocessing=default_preprocessing, dirpath=None):

        self._cdcfileinfo, self._cdcweb, self._cdcfiles = self._cdc_files_info()
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','cdc_data')

        self._csv_path = os.path.join(self._dirpath, 'csv')
        if not os.path.exists(self._dirpath):
            os.mkdir(self._dirpath)

        for f in self._cdcfiles:
            try:
                filename = os.path.join(self._dirpath, f)
                if not os.path.exists(filename):
                    print("Downloading file {}".format(f))
                    file = requests.get(os.path.join(self._cdcweb, f), allow_redirects=True)
                    fp = open(filename, 'wb')
                    fp.write(file.content)
                    fp.close()                    
            except IOError as err:
                print("IOError: {}".format(err))
                sys.exit(1)

        self._convert_xpt_to_csv()
        #if custom_preprocessing:
        #    self._data = custom_preprocessing(df)


    def _cdc_files_info(self):
        # List of files (i.e. questionnaires) in the CDC dataset. The following 4 files were ignored due to processing issues.
        # RXQ_RX_H: Prescription Medications
        # SMQ_H: Smoking - Cigarette Use
        # PUQMEC_H: Pesticide Use
        # RXQ_DRUG.xpt: Prescription Medications - Drug Information
        # If the errors can be fixed, they can be added to this list.

        cdcfileinfo = ['Acculturation', 'Alcohol Use', 'Blood Pressure & Cholesterol', 'Cardiovascular Health',
                    'Cognitive Functioning', 'Consumer Behavior', 'Creatine Kinase', 'Current Health Status',
                    'Dermatology', 'Diabetes', 'Diet Behavior & Nutrition', 'Disability', 'Drug Use', 'Early Childhood',
                    'Food Security', 'Health Insurance', 'Hepatitis', 'Hospital Utilization & Access to Care',
                    'Housing Characteristics', 'Immunization', 'Income', 'Kidney Conditions - Urology',
                    'Medical Conditions', 'Mental Health - Depression Screener',
                    'Occupation', 'Oral Health', 'Osteoporosis', 'Physical Activity',
                    'Physical Functioning', 'Preventive Aspirin Use','Reproductive Health',
                    'Sexual Behavior', 'Sleep Disorders', 'Smoking - Household Smokers', 'Smoking - Recent Tobacco Use',
                    'Smoking - Secondhand Smoke Exposure', 'Taste & Smell', 'Volatile Toxicant (Subsample)',
                    'Weight History', 'Weight History - Youth'
                   ]
        cdcweb = 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/'
        cdcfiles = ["ACQ_H.XPT",
        "ALQ_H.XPT",
        "BPQ_H.XPT",
        "CDQ_H.XPT",
        "CFQ_H.XPT",
        "CBQ_H.XPT",
        "CKQ_H.XPT",
        "HSQ_H.XPT",
        "DEQ_H.XPT",
        "DIQ_H.XPT",
        "DBQ_H.XPT",
        "DLQ_H.XPT",
        "DUQ_H.XPT",
        "ECQ_H.XPT",
        "FSQ_H.XPT",
        "HIQ_H.XPT",
        "HEQ_H.XPT",
        "HUQ_H.XPT",
        "HOQ_H.XPT",
        "IMQ_H.XPT",
        "INQ_H.XPT",
        "KIQ_U_H.XPT",
        "MCQ_H.XPT",
        "DPQ_H.XPT",
        "OCQ_H.XPT",
        "OHQ_H.XPT",
        "OSQ_H.XPT",
        "PAQ_H.XPT",
        "PFQ_H.XPT",
        "RXQASA_H.XPT",
        "RHQ_H.XPT",
        "SXQ_H.XPT",
        "SLQ_H.XPT",
        "SMQFAM_H.XPT",
        "SMQRTU_H.XPT",
        "SMQSHS_H.XPT",
        "CSQ_H.XPT",
        "VTQ_H.XPT",
        "WHQ_H.XPT",
        "WHQMEC_H.XPT"]

        return cdcfileinfo, cdcweb, cdcfiles

    def _convert_xpt_to_csv(self):
        if not os.path.exists(self._csv_path):
            os.mkdir(self._csv_path)

        for i in range(len(self._cdcfiles)):
            f = self._cdcfiles[i]
            finfo = self._cdcfileinfo[i]
            xptfile = os.path.join(self._dirpath, f)
            csvfile = os.path.join(self._csv_path, f)
            csvfile = os.path.splitext(csvfile)[0]
            csvfile = csvfile + ".csv"
            if not os.path.exists(csvfile):
                print("converting ", finfo, ": ", xptfile, " to ", csvfile)
                with open(xptfile, 'rb') as in_xpt:
                    with open(csvfile, 'w',newline='') as out_csv:
                        reader = xport.Reader(in_xpt)
                        writer = csv.writer(out_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(reader.fields)
                        for row in reader:
                            writer.writerow(row)


    def get_csv_file(self, filename):
        return pd.read_csv(os.path.join(self._csv_path, filename))

    def get_csv_file_names(self):
        return [os.path.splitext(x)[0]+".csv" for x in self._cdcfiles]
