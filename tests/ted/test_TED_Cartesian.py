import unittest
import os
import pandas as pd    
from sklearn import svm
from sklearn.model_selection import train_test_split

from aix360.algorithms.ted import TED_CartesianExplainer
from aix360.datasets.ted_dataset import TEDDataset

class TestTED_Cartesian(unittest.TestCase):
   """
   A class to test the TED_Cartesian explainer
   """

   def setUp(self):
      """
      Perform setup code of opening datafile, constructing train/test split, created TED_CartesianExplainer and
      """

      # Decompose the dataset into X, Y, E     
      X, Y, E = TEDDataset().load_file('Retention.csv')               

      # set up train/test split
      self.X_train, self.X_test, self.Y_train, self.Y_test, self.E_train, self.E_test = train_test_split(X, Y, E, test_size=0.20, random_state=0)

      # Create classifier and pass to TED_Cartesian
      self.estimator = svm.SVC(kernel='linear')   
      self.ted = TED_CartesianExplainer(self.estimator)
      self.ted.fit(self.X_train, self.Y_train, self.E_train)   # train classifier

   def test_instances(self):
      """
      Test 2 instances from the test dataset (hard-coded here) to ensure they give the correct Y and E value.
      Test all 3 interfaces to TED_Cartesian:  predict_explain, predict, and explain
      """

      X1 = [[1, 2, -11, -3, -2, -2,  22, 22]]
      # correct answers:  Y:0; E:3
      Y, E = self.ted.predict_explain(X1)
      self.assertEqual(Y, -10)
      self.assertEqual(E, 13)

      Y = self.ted.predict(X1)
      self.assertEqual(Y, -10)

      E = self.ted.explain(X1)
      self.assertEqual(E, 13)

      X2 = [[3, 1, -11, -2, -2, -2, 296, 0]]
      ## correct answers: Y:-11, E:25
      Y, E = self.ted.predict_explain(X2)
      self.assertEqual(Y, -11)
      self.assertEqual(E, 25)

      Y = self.ted.predict(X2)
      self.assertEqual(Y, -11)

      E = self.ted.explain(X2)
      self.assertEqual(E, 25)

   def test_score(self): 
      """
      This test evaluates the accuracy (in 3 ways) of the TED-enhanced classifier on the full test suite.
      It ensures the accuracy is above the correct values for this classifier.
      NOTE: accuracy will depend on the classifier used.
      """
      YE_accuracy, Y_accuracy, E_accuracy = self.ted.score(self.X_test, self.Y_test, self.E_test)    # evaluate the classifier

      # Expected Accuracy for this dataset and classifier
      # Accuracy of YE = 0.8510
      # Accuracy of Y = 0.8615
      # Accuracy of E = 0.8510

      self.assertGreaterEqual(YE_accuracy, 0.85)
      self.assertGreaterEqual(Y_accuracy, 0.85)
      self.assertGreaterEqual(E_accuracy, 0.85)

if __name__ == '__main__':
    unittest.main()
