# This file will generate a synthetic dataset to predict employee attrition
# Like most datasets it will have a feature vector and a Y label for each instance.
# However, unlike most datasets it will also have an Explanation (E) for each instance, encoded as an non-negative integer.
# This is motivated by the TED framework, but can be used by other explainability algorithms as a metric for explainability
# See the AIES'19 paper by Hind et al for more information on the TED framework.
# See the tutorial notebook TED_Cartesian_test for information about how to use this dataset and the TED framework.
# The comments in this code also provide some insight into how this dataset is generated

import random
from random import choices
import pandas as pd

Any = -99   # This is only applicable in the rule
Low = -1    # These 3, Low, Med, High, can be values in the dataset and are used in the rules
Med = -2
High = -3
Yes = -10   # This is the positive Y label
No = -11    # This is the negative Y label
Random = -12  # This signifies a random choice should be made for the Y label (either Yes or No)                        ]

# Features, values, and distribution, details below
featureThresholds = [
    # 1 Position:  4(5%), 3(20%), 2(30%), 1(45%)
    [4, [0.05, 0.20, 0.30, 0.45]],

    # 2 Organization "Org": 3(30%); 2(30%); 1(40%)
    [3, [0.30, 0.30, 0.40]],

    # 3 Potential "Pot": Yes (50%), No (50%)
    [2, [0.50, 0.50]],

    # 4 Rating value "Rat": High(15%), Med(80%), Low(5%)
    [3, [0.15, 0.80, 0.05]],

    # 5 Rating Slope "Slope": High (15%), Med(80%), Low(5%)
    [3, [0.15, 0.80, 0.05]],

    # 6 Salary Competitiveness "Sal": High (10%); Med(70%); Low(20%)
    [3, [0.10, 0.70, 0.20]],

    # 7 Tenure Low "TenL" & High Values "TenH":  [0..360], 30% in 0..24; 30% in 25..60; 40% in 61..360
    [3, [0.30, 0.30, 0.40], [[0, 24], [25, 60], [61, 360]]],

    # 8 Position Tenure Low "BTenL" & High Values "BTenH": [0..360],  70% in 0..12; 20% in 13..24; 10% in 25..360
    #    Position tenure needs to be lower than tenure, ensured in generation code below
    [3, [0.70, 0.20, 0.10], [[0, 12], [13, 24], [25, 360]]]
]

# Some convenient population lists
HighMedLowPopulation = [High, Med, Low]
YesNoPopulation = [Yes, No]
Index3Population = [0, 1, 2]
Integer4Population = [4, 3, 2, 1]
Integer3Population = [3, 2, 1]

# Rules used to label a feature vector with a label and an explanation
# Format: features, label, explanation #, Explanation String 
RetentionRules = [ 
    #POS  ORG    Pot  RAT Slope  SALC  TENL H       BTEN LH  
    [Any, 1,     Any, High,  Any,	Low,  Any, Any,   Any, Any, #0
        Yes, 2, "Seeking Higher Salary in Org 1"],
    [1,   1,	  Any, Any, Any,	Any,  Any, Any,   15,  Any,	#1
        Yes, 3, "Promotion Lag, Org 1, Position 1"],
    [2,   1,	  Any, Any, Any,	Any,  Any, Any,   15,  Any,	#2
        Yes, 3, "Promotion Lag, Org 1, Position 2"],
    [3,   1,	  Any, Any, Any,	Any,  Any, Any,   15,  Any,	#3
        Yes, 3, "Promotion Lag, Org 1, Position 3"],
    [1,   2,	  Any, Any, Any,	Any,  Any, Any,   20,  Any,	#4
        Yes, 4, "Promotion Lag, Org 2, Position 1"],
    [2,   2,	  Any, Any, Any,	Any,  Any, Any,   20,  Any,	#5
        Yes, 4, "Promotion Lag, Org 2, Position 2"],
    [3,   2,      Any, Any, Any,	Any,  Any, Any,   30,  Any,	#6
        Yes, 5, "Promotion Lag, Org 2, Position 3"],
    [1,   3,      Any, Any, Any,	Any,  Any, Any,   20,  Any,	#7
        Yes, 6, "Promotion Lag, Org 3, Position 1"],
    [2,   3,	  Any, Any, Any,	Any,  Any, Any,   30,  Any,	#8
        Yes, 7, "Promotion Lag, Org 3, Position 2"],
    [3,   3,	  Any, Any, Any,	Any,  Any, Any,   30,  Any,	#9
        Yes, 7, "Promotion Lag, Org 3, Position 3"],
    [1,   1,      Any, Any, Any,	Any, 0,    12,   Any, Any,	#10
        Yes, 8, "New employee, Org 1, Position 1"],
    [2,   1,      Any, Any, Any,	Any, 0,    12,   Any, Any,	#11
        Yes, 8, "New employee, Org 1, Position 2"],
    [3,   1,      Any, Any, Any,	Any, 0,    30,   Any, Any,	#12
        Yes, 9, "New employee, Org 1, Position 3"],
    [1,   2,      Any, Any, Any,	Any, 0,    24,   Any, Any,	#13
        Yes, 10, "New employee, Org 2, Position 1"],
    [2,   2,      Any, Any, Any,	Any, 0,    30,   Any, Any,	#14
        Yes, 11, "New employee, Org 2, Position 2"],
    [Any, 1,      Any, Low, High, Any,    Any, Any,   Any, Any,	#15
        Yes, 13, "Disappointing evaluation, Org 1"],
    [Any, 2,      Any, Low, High, Any,    Any, Any,   Any, Any,	#16
        Yes, 14, "Disappointing evaluation, Org 2"],
    [Any, Any, Yes, Med, High, Low,    Any, Any,   Any, Any,	#17
        Yes, 15, "Compensation doesn't match evaluations, Med rating"],
    [Any, Any, Yes, High, High, Low,   Any, Any,  Any, Any,	#18
        Yes, 15, "Compensation doesn't match evaluations, High rating"],
    [Any, 1,   Yes, Med, High, Med,    Any, Any,   Any, Any,	#19
	    Yes, 16, "Compensation doesn't match evaluations, Org 1, Med rating"],
    [Any, 2,   Yes, Med, High, Med,    Any, Any,   Any, Any,	#20
	    Yes, 16, "Compensation doesn't match evaluations, Org 2, Med rating"],
    [Any, 1,   Yes, High, High, Med,   Any, Any,   Any, Any,	#21
	    Yes, 16, "Compensation doesn't match evaluations, Org 1, High rating"],
    [Any, 2,   Yes, High, High, Med,   Any, Any,   Any, Any,	#22
	    Yes, 16, "Compensation doesn't match evaluations, Org 2, High rating"],
    [Any, 1,   Any, Any, Med,	Med, 120, 180,   Any, Any,	#23
	    Yes, 17, "Mid-career crisis, Org 1"],
    [Any, 2,   Yes, Any, Any,	Med, 130, 190,   Any, Any,	#24
	    Yes, 18, "Mid-career crisis, Org 2"]
]

def ruleValToString(val):
    """ Convert the value passed into a string """
    if val == Any :
        return "Any"
    elif val == Low :
        return "Low"
    elif val == Med :
        return "Med"
    elif val == High :
        return "High"
    elif val == Yes :
        return "Yes"
    elif val == No :
        return "No"
    elif val == Random :
        return "Random"
    else :
        return str(val)

def printFeatureStringHeader() :
    """ Print the feature headings """
    print("     Feature Headings")
    print("[Pos, Org, Pot, Rating, Slope, Salary Competitiveness, Tenure, Position Tenure]")
         
def featuresToString(featureVector) :
    """ Convert a feature vector into is string format"""
    val = "["
    for i in range(0, 2) :   # These features are just ints, Position, Organization
        val += str(featureVector[i])
        val += " "           
    for i in range(2, 6) :   # show encoding for these: Potential, Rating, Rating Slope, Salary Competitiveness
        val += ruleValToString(featureVector[i]) 
        val += " "
    for i in range(6, 8) :   # These features are just ints: Tenure and Position Tenure
        val += str(featureVector[i])
        val += " "           
    val += "]"
    return val

def printRule(rule) :
    """ Print the passed rule """
    print("Rule: ", end='')
    for i in rule[0:1]:   # ints or Any: Position and Organization
        if i == Any:
           print(ruleValToString(i) + ", ", end='')

    for i in rule[2:5]:   # encoded: Potentional, Rating, Rating Slope, Salary Competitiveness
        print(ruleValToString(i) + ", ", end='')

    for i in rule[6:9]:   # next 4 are ints or ANY: Tenure Low, Tenure High, Position Tenure Low, Position Tenure High
        if i == Any :
            print(ruleValToString(i) + ", ", end='')
        else :
            print(str(i) + ", ", end='')       
    print("==> "+ ruleValToString(rule[10]) + "[" + str(rule[11]) + "] " + str(rule[12]))

def printRules(rules) :
    """ print all rules"""
    for r in rules:
        printRule(r)

########################################################################

def chooseRangeValue(thresholds, rangeList):
    """  Generate a random value based on the probability weights (thresholds) and list of ranges passed
    Args: 
        thresholds : list of probabilities for each choice
        rangeList: a list of pair lists giving the lower and upper bounds to choose value from 
    """

    # pick a number 1..3 from weights
    rangeVal = choices(Index3Population, thresholds)

    # get the appropriate range given rangeVal
    interval = rangeList[rangeVal[0]]

    # construct a population list from the result
    intervalPopulation = list(range(interval[0], interval[1]))

    # construct a equally prob weights list
    numElements = interval[1] - interval[0]
    probVal = 1.0 / numElements
    probList = [probVal] * numElements

    # now choose the value from the population based on the weights
    val = choices(intervalPopulation, probList)
    return val[0]


def chooseValueAndAppend(instance, population, weights) :
    """ Choose a random value from the population using weights list and append it to the passed instance
    """
    val = choices(population, weights)
    instance.append(val[0])

def generateFeatures(numInstances) :
    """ generate the features (X) values for the dataset
    Args:
        numInstances (int) : number of instances to genreate
    Returns:
        dataset (list of lists) : the dataset with features, but no labels or explanations yet
    """
    assert(numInstances > 0)

    dataset = []
    for i in range(numInstances) :
        instance = []

        #POS  ORG    Pot  Rating Slope  SALC  TENL H       BTEN LH  
        chooseValueAndAppend(instance, Integer4Population, featureThresholds[0][1])   # Position
        chooseValueAndAppend(instance, Integer3Population, featureThresholds[1][1])   # Org
        chooseValueAndAppend(instance, YesNoPopulation,    featureThresholds[2][1])   # Potential
        chooseValueAndAppend(instance, HighMedLowPopulation, featureThresholds[3][1]) # Rating
        chooseValueAndAppend(instance, HighMedLowPopulation, featureThresholds[4][1]) # Rating slope
        chooseValueAndAppend(instance, HighMedLowPopulation, featureThresholds[5][1]) # Sal competitiveness

        val1 = chooseRangeValue(featureThresholds[6][1], featureThresholds[6][2]) # Tenure
        instance.append(val1)

        # Position tenure needs to be <= Tenure
        val2 = chooseRangeValue(featureThresholds[7][1], featureThresholds[7][2]) # Pos Tenure
        if val2 > val1 :
            val2 = val1
        instance.append(val2)
        dataset.append(instance)
    
    return dataset

#####################################################################################################

def match(ruleVal, featureVal) :
    """ Check if passed ruleVal matches the featureVal or if ruleVal is Any, which matches everything 
    """

    # print("Match called: "+ ruleValToString(ruleVal) + " " + ruleValToString(featureVal))
    if ruleVal == Any :
        return True
    return (ruleVal == featureVal)

def intervalMatch(ruleValLower, ruleValUpper, featureVal) :
    """ Check to see if featureVal is in the interval defined by [ruleValLower, ruleValUpper)
    """

    # Any in lower bound matches all values, (upper bound doesn't matter)
    if ruleValLower == Any :
        return True

    if ruleValLower <= featureVal :
        # Any in upper bound means infinitity
        if featureVal < ruleValUpper or ruleValUpper == Any :
            return True
    
    return False

def ruleMatch(rule, featureVector) :
    """  Determine if the passed featureVector matches the passed rule 
    """
    if (False) :
        print("ruleMatch called, ", end="")
        printRule(rule)
        print(" feature vector: " + featuresToString(featureVector) )

    for i in range(0, 6) :            # loop over first 6 features, 0..5
        if not match(rule[i], featureVector[i]) :   # if we don't find a feature match, the rule doesn't match
            # print("Didn't match feature #", i, ruleValToString(featureVector[i]))
            return False
    
    # These features are interval-based, so need a different matching routine
    if not intervalMatch(rule[6], rule[7], featureVector[6]) :  # rule[6] and rule[7] have the lower and upper bounds of interval
        # print("Didn't match feature # 6: ", featureVector[6])
        return False
    if not intervalMatch(rule[8], rule[9], featureVector[7]) :  # rule[8] and rule[9] have the lower and upper bounds of interval
        # print("Didn't match feature # 7: ", featureVector[7])
        return False
   
    # print("Matched all features")
    return True                                     # if we didn't find a non-match by now, we found a match

def findRule(instance, ruleSet) :
    """ find the rule(s) that matches the feture vector passed
    """

    # print("*Looking for rule match for Feature vector: " + featuresToString(instance))
    ruleNumber = 0      # counter to track rule number
    ruleMatches = []    # will hold all rule numbers that matched
    for rule in ruleSet :
        if (ruleMatch(rule, instance)) :
            ruleMatches.append(ruleNumber)
            counts[ruleNumber] += 1               # update global histogram of rule matches for stats reporting

            if (False) :
                print(" ruleMatch found at rule #" + str(ruleNumber))
                print(" ", end="")
                printRule(rule)

        ruleNumber += 1

    return ruleMatches

def countAnys(rule) :
    """ Count the number of Anys in the passed rule.  An "Any" is a wildcard that matches all values
    """
    count = 0
    for feature in RetentionRules[rule] :
        if feature == Any :
            count += 1

    return count

def pickBestRule(ruleList) :
    """ Choose the rule with the least number of Any's in it
    """
    assert(len(ruleList) > 0)

    # print("ruleList: ", ruleList)
    minAnys = len(RetentionRules[0]) + 1      # initialize to a value larger than possible # of Anys in a rule
    bestRule = -1
    for rule in ruleList :
        # Count # of Any's in rule # rule
        count = countAnys(rule)
        if count < minAnys :
            minAnys = count
            bestRule = rule

    assert(bestRule != -1)     # We should find a best rule
    return bestRule

def addLabelsAndExplanations(dataset, rules) :
    """ This function will use a ruleset to add labels (Y) and explanations/rules (E) to a passed dataset
    Arg:
        dataset (list of lists) : a list of feature vectors (list)
        rules (list of lists) : a list of rules
    """

    noMatches = 0                 # Counters to record how often there are no (Yes) matches, 1 (Yes) match, and multiple (Yes) matches
    multiMatches = 0
    oneMatches = 0
    for instance in dataset :
        ruleMatches = findRule(instance, rules)

        if len(ruleMatches) == 0 :     # We didn't match a (Yes) rule, so this ia No situation
            rule = NoRiskRuleNum
            label = No
            noMatches +=1
        elif len(ruleMatches) > 1 :    # Matched multiple Yes rules, need to pick one
            rule = pickBestRule(ruleMatches)
            assert(rule >= 0 and rule < len(rules))   # Ensure rule number is valid
            label = Yes
            multiMatches += 1
        else :                         # Found 1 Yes rule match, it's the winner
            rule = ruleMatches[0]
            label = Yes
            oneMatches += 1
            assert(rule >= 0 and rule < len(rules))   # Ensure rule number is valid

        # print("Label: " + ruleValToString(label) + ", Rule: " + ruleValToString(rule))

        instance.append(label)
        instance.append(rule)   # add the label and explanation (rule #) to the featureVector

    if (True) :
        print("\nRule matching statistics: ")
        totalYes = oneMatches + multiMatches
        total = oneMatches + multiMatches + noMatches
        print("  Yes Labels: {}/{} ({:.2f}%)".format(totalYes, total, totalYes/total*100))
        print("    Matched 1 Yes rule: {}/{} ({:.2f}%)".format(oneMatches, totalYes, oneMatches/totalYes*100))
        print("    Matched multiple Yes rules: {}/{} ({:.2f}%)".format(multiMatches, totalYes, multiMatches/totalYes*100))
        print("  No Laels: {}/{} ({:.2f}%)".format(noMatches, total, noMatches/total*100))

def printRuleUsage(counts, total) :
    print("\nHistogram of rule usage:")
    ruleNum = 0
    for num in counts :
        print(" Rule {} was used {} times, {:.2f}%".format(ruleNum, num, num/total*100))
        ruleNum += 1

        
numRentionRules = len(RetentionRules)
counts = [0]*numRentionRules
NoRiskRuleNum = numRentionRules    # the No Risk to leave rule is 1 more than than the total rules [0..]

random.seed(1)
# printFeatureStringHeader()
numInstances = 10000
dataset = generateFeatures(numInstances)

addLabelsAndExplanations(dataset, RetentionRules)

printRuleUsage(counts, numInstances)

# insert TED headers
NumFeatures = len(featureThresholds)
header = list(range(NumFeatures))
header.append("Y")
header.append("E")
dataset.insert(0, header)

# write to csv file
my_df = pd.DataFrame(dataset)
my_df.to_csv('Retention.csv', index=False, header=False)
