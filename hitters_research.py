################################################
# End-to-End Hitters Machine Learning Pipeline I
################################################


# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import missingno as msno
pd.set_option("display.max_rows",None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

################################################
# 1. Exploratory Data Analysis
################################################

#Functions:

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    return dataframe

def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def edit_outlier(dataframe,variable,q3=0.90):
    #quartile1 = dataframe[variable].quantile(q1)
    #quartile3 = dataframe[variable].quantile(q3)
    variable_up = int(dataframe[variable].quantile(q3))
    dataframe = dataframe[(dataframe[variable] < variable_up)]
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

df = pd.read_csv("HAFTA_07/Ders Notları/hitters.csv")
df.head()



# Quick review
check_df(df)

# Separate variable types
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

# reviewing categorical variables
for col in cat_cols:
    cat_summary(df,col)

# reviewing numeric variables
df[num_cols].describe().T
#for col in num_cols:
    #num_summary(df,col)

# see correlations between numeric variables
correlation_matrix(df,num_cols)

# review numerical variables with target
for col in num_cols:
    target_summary_with_num(df, "Salary", col)

# review numerical variables with target
for col in cat_cols:
    target_summary_with_cat(df,"Salary",col)

# visualize missing values
msno.matrix(df)
plt.show()

# checking categorical variable whether there is any difference or not between values of them
df.groupby("Division").agg({"Salary" : "mean"})


# target dependence variable has null values. It may cause to model prediction mistakes. so we need imputation to target null's.
df.head()
df.isnull().sum()
data = df.copy()
data.isnull().sum()

# some categorical values are in dataset, it has to be ready to modelling. so we'll encode them as numerical.
# before doing that, we'll check difference between leagues and their impact on salaries.
# first change on dataset is drop na's.

data.dropna(inplace=True)
data.isnull().sum()
data.shape

# > AB Testing

# >> Is there any significant difference statistically between division E and division W?

# >>> Normalization Assumption
# "H0: Normal distribution
# "H1: Not normal distribution
# pvalue < 0.05 than H0 is rejected. Because of that we don't need to look variance homogenity.

test_stat, pvalue = shapiro(data.loc[data["Division"] == "E", "Salary"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(data.loc[data["Division"] == "W", "Salary"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# >>> Assumption are not related. So non-parametric test will be used.

test_stat, pvalue = mannwhitneyu(data.loc[data["Division"] == "E", "Salary"],
                           data.loc[data["Division"] == "W", "Salary"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# 0.0087
# HO: M1 = M2
# H1: M1 != M2
# pvalue < 0.05 then HO hypothesis is rejected
# pvalue > 0.05 then HO hypothesis is not rejected

# Result means there is significant difference between divisions about effecting target variable.
# The conclusion is division E and division W salaries are different from each other.
# Encoding should be include ordinality effects.

data = df.copy()
data["Division"].value_counts()

data.head()
check_df(data)

# >> Label Encoding
label_encoder(data,"Division")

# w , division encode 1
# e , division encode 0

# >> One Hot Encoding
data = one_hot_encoder(data,["League","NewLeague"])

# >> Robust Scaler
cat_cols, num_cols, cat_but_car = grab_col_names(data, cat_th=5, car_th=20)
num_cols = [col for col in num_cols if "Salary" not in col]
rb = RobustScaler()
data[num_cols] = rb.fit_transform(data[num_cols])

# Prediction model for null values
test_data = data[~data["Salary"].isnull()]
test_data.isnull().sum()
len(test_data)

X = test_data.drop("Salary",axis=1)
y = test_data["Salary"]

# model testing (RMSE)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)
y_pred_df = pd.DataFrame({"y_pred": y_pred})
np.sqrt(mean_squared_error(y_test, y_pred)) # 365

np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error"))) #331

# main Salary null values imputating with model based predicted values

null_data = data[data["Salary"].isnull()]
null_data.shape[0]
null_data.index

X = null_data.drop("Salary",axis=1)
y = null_data["Salary"]
X.shape
y.shape

y_null_pred = reg_model.predict(X)
y_null_pred_df = pd.DataFrame({"y_null_pred": y_null_pred},index=null_data.index)
y_null_pred_df


df2 = df.copy() # index check

# imputating calculated values to null values #####

df.loc[df["Salary"].isnull() == True,"Salary"] = y_null_pred_df.values
df.isnull().sum()
df.shape

################################################
# 2. Data Preprocessing & Feature Engineering
################################################


# Convert columns to uppercase
df.columns = [col.upper() for col in df.columns]

# Creating new features
"accurate hits in 1 season 86-87"
df["NEW_ACCHIT_OS"] = df["HITS"] / df["ATBAT"]

"conversion rate of hits to points"
df["NEW_CONR_HTP"] = df["RUNS"] / df["HITS"]

"running oppenent player"
df["NEW_RUN_OPP_PL"] = df["RBI"] / df["ATBAT"]

"score contribution in 1 season 86-87"
df["NEW_SCO_CONT"] = df["RUNS"] * 0.6 + df["ASSISTS"] * 0.4

"golden scores"
df["NEW_GOLD_SCO"] = df["HMRUN"] / df["ATBAT"]

"effect on oppenent players"
df["NEW_EFF_OPP"] = df["RBI"] * df["WALKS"]

"corr between errors and assists"
df["NEW_ERR_AS"] = df["ERRORS"] * df["ASSISTS"]

"86-97 season hit performance score comparing to whole carreer"
df["NEW_SEASONAL_HITS"] = df["ATBAT"] / df["CATBAT"]

"86-97 season hit accuration performance score comparing to whole carreer"
df["NEW_SEASONAL_HITS"] = df["ATBAT"] / df["CATBAT"]

"86-97 season hit count performance score comparing to whole carreer"
df["NEW_SEASONAL_HITCOUNTS"] = df["HITS"] / df["CHITS"]

"86-97 season gaining points to team performance comparing to whole carreer"
df["NEW_SEASONAL_GAINPOINTS"] = df["RUNS"] / df["CRUNS"]

"86-97 season number of players that run performance comparing to whole carreer"
df["NEW_SEASONAL_PLAYERSRUN"] = df["RBI"] / df["CRBI"]

"86-97 season number of mistakes made by the opposing player performance comparing to whole carreer"
df["NEW_SEASONAL_NUMOFMIS"] = df["WALKS"] / df["CWALKS"]


check_df(df)

# getting seperated type of values
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

# Checking outliers in numeric cols
for col in num_cols:
    print(col, check_outlier(df, col))

# Replacing outliers with thresholds
for col in num_cols:
    replace_with_thresholds(df, col)

# Checking outliers again, outliers must be changed with quartiles
for col in num_cols:
    print(col, check_outlier(df, col))

df.isnull().sum()

check_outlier(df,num_cols)
check_outlier(df,cat_cols)
check_outlier(df,"SALARY")


df["SALARY"].describe([.05,.10,.25,.35,.45,.50,.60,.70,.80,.85,.90,.95,.97,.99]).T

# Assignment mean values of variables to na's
df["NEW_SEASONAL_PLAYERSRUN"] = df["NEW_SEASONAL_PLAYERSRUN"].fillna(df["NEW_SEASONAL_PLAYERSRUN"].mean())
df["NEW_SEASONAL_NUMOFMIS"] = df["NEW_SEASONAL_NUMOFMIS"].fillna(df["NEW_SEASONAL_NUMOFMIS"].mean())



df.groupby("LEAGUE")["SALARY"].agg({"mean"})

df.groupby("DIVISION")["SALARY"].agg({"mean"})

df.groupby("NEWLEAGUE")["SALARY"].agg({"mean"})


# Checking updated dataframe
check_df(df)

# Label Encoding for binary variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# One Hot Encoding for remaining variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

num_cols = [col for col in num_cols if "SALARY" not in col]

# Standardization with Robust Scaler
rb = RobustScaler()

df[num_cols] = rb.fit_transform(df[num_cols])

df.head()

# Target and independent variables creating
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

check_df(df)

# after than, we turns to function all we did above.

def hitters_data_prep(df):
    df.columns = [col.upper() for col in df.columns]

    df["NEW_ACCHIT_OS"] = df["HITS"] / df["ATBAT"]
    df["NEW_CONR_HTP"] = df["RUNS"] / df["HITS"]
    df["NEW_RUN_OPP_PL"] = df["RBI"] / df["ATBAT"]
    df["NEW_SCO_CONT"] = df["RUNS"] * 0.6 + df["ASSISTS"] * 0.4
    df["NEW_GOLD_SCO"] = df["HMRUN"] / df["ATBAT"]
    df["NEW_EFF_OPP"] = df["RBI"] * df["WALKS"]
    df["NEW_ERR_AS"] = df["ERRORS"] * df["ASSISTS"]
    df["NEW_SEASONAL_HITS"] = df["ATBAT"] / df["CATBAT"]
    df["NEW_SEASONAL_HITS"] = df["ATBAT"] / df["CATBAT"]
    df["NEW_SEASONAL_HITCOUNTS"] = df["HITS"] / df["CHITS"]
    df["NEW_SEASONAL_GAINPOINTS"] = df["RUNS"] / df["CRUNS"]
    df["NEW_SEASONAL_PLAYERSRUN"] = df["RBI"] / df["CRBI"]
    df["NEW_SEASONAL_NUMOFMIS"] = df["WALKS"] / df["CWALKS"]
    df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "A"), "NEW_LOYALPLAYERS"] = "STABLE"
    df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "N"), "NEW_LOYALPLAYERS"] = "STABLE"
    df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "N"), "NEW_LOYALPLAYERS"] = "TEAMCHANGED"
    df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "A"), "NEW_LOYALPLAYERS"] = "TEAMCHANGED"

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    for col in num_cols:
        replace_with_thresholds(df, col)

    df["NEW_SEASONAL_PLAYERSRUN"] = df["NEW_SEASONAL_PLAYERSRUN"].fillna(df["NEW_SEASONAL_PLAYERSRUN"].mean())
    df["NEW_SEASONAL_NUMOFMIS"] = df["NEW_SEASONAL_NUMOFMIS"].fillna(df["NEW_SEASONAL_NUMOFMIS"].mean())

    df = df.dropna(subset=['SALARY'])

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]

    for col in binary_cols:
        df = label_encoder(df, col)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

    one_hot_encoder(df, ohe_cols)

    num_cols = [col for col in num_cols if "SALARY" not in col]

    rb = RobustScaler()

    df[num_cols] = rb.fit_transform(df[num_cols])

    y = df["SALARY"]
    X = df.drop(["SALARY"], axis=1)

    return X, y


X, y = hitters_data_prep(df)


















