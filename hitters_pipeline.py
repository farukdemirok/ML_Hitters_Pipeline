################################################
# End-to-End Hitters Machine Learning Pipeline II
################################################

# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, train_test_split, validation_curve, learning_curve
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.impute import KNNImputer

df = pd.read_csv("HAFTA_07/Ders Notları/hitters.csv")

def hitters_data_prep(df):
    data = df.copy()

    # >> Label Encoding
    label_encoder(data, "Division")

    # w , division encode 1
    # e , division encode 0

    # >> One Hot Encoding
    data = one_hot_encoder(data, ["League", "NewLeague"])

    # >> Robust Scaler
    cat_cols, num_cols, cat_but_car = grab_col_names(data, cat_th=5, car_th=20)
    num_cols = [col for col in num_cols if "Salary" not in col]
    rb = RobustScaler()
    data[num_cols] = rb.fit_transform(data[num_cols])

    # Prediction model for null values
    test_data = data[~data["Salary"].isnull()]

    X = test_data.drop("Salary", axis=1)
    y = test_data["Salary"]

    # model testing (RMSE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    y_pred = reg_model.predict(X_test)
    y_pred_df = pd.DataFrame({"y_pred": y_pred})

    # main Salary null values imputating with model based predicted values

    null_data = data[data["Salary"].isnull()]

    X = null_data.drop("Salary", axis=1)
    y = null_data["Salary"]


    y_null_pred = reg_model.predict(X)
    y_null_pred_df = pd.DataFrame({"y_null_pred": y_null_pred}, index=null_data.index)

    # imputating calculated values to null values #####

    df.loc[df["Salary"].isnull() == True, "Salary"] = y_null_pred_df.values

    df.columns = [col.upper() for col in df.columns]

    df["NEW_PRECIOUS_SCO"] = df["CHMRUN"] / df["HITS"]
    df["NEW_AVE_HITS"] =  df["CHITS"] / df["YEARS"]
    df["NEW_AVE_WALKS"] = df["CWALKS"] / df["YEARS"]
    df["NEW_AVE_RUNS"] = df["CRUNS"] / df["YEARS"]
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
    df["NEW_EXP"] = df["YEARS"] * df["PUTOUTS"]

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    for col in num_cols:
        col,check_outlier(df,num_cols)

    for col in num_cols:
        replace_with_thresholds(df, col)

    df.isnull().sum()
    df.dropna(inplace=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    ordinal_variables = ["DIVISION"]
    nominal_unwanted_variables = ["LEAGUE,NEWLEAGUE"]
    #binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

    df.drop(nominal_unwanted_variables,inplace=True,axis=1)

    for col in ordinal_variables:
        df = label_encoder(df, col)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() >= 2 and "DIVISION" not in col]

    "categorical cols are not associated with target, very few effects on it, by this cause we can drop specified cols as indicated"

    df = one_hot_encoder(df, cat_cols)

    num_cols = [col for col in num_cols if "SALARY" not in col]

    rb = RobustScaler()

    df[num_cols] = rb.fit_transform(df[num_cols])

    y = df["SALARY"]
    X = df.drop(["SALARY"], axis=1)

    return X, y


X, y = hitters_data_prep(df)


######################################################
# 3. Base Models
######################################################

def base_models(X, y, scoring="f1"):
    print("Base Models....")
    models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          # ("CatBoost", CatBoostRegressor(verbose=False))
          ]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")


base_models(X, y, scoring="f1")

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


# list feature importances for a regressor model like LGBM
pre_model = LGBMRegressor(random_state=17).fit(X, y)
feature_imp = pd.DataFrame({'Feature': X.columns, 'Value': pre_model.feature_importances_})
feature_imp.sort_values("Value", ascending=False)

"Show sorted values according to one LightGBM that one of the best model before make hyperparameter optimization"
pre_model1 = LGBMRegressor().fit(X, y)
feature_imp1 = pd.DataFrame({'Feature': X.columns, 'Value': pre_model1.feature_importances_})
feature_imp1.sort_values("Value", ascending=False)
#plot_importance(pre_model1, X)

"Show sorted values according to one Random Forest that one of the best model before make hyperparameter optimization"
pre_model2 = RandomForestRegressor().fit(X, y)
feature_imp2 = pd.DataFrame({'Feature': X.columns, 'Value': pre_model2.feature_importances_})
feature_imp2.sort_values("Value", ascending=False)
#plot_importance(pre_model2, X)

"Show sorted values according to one Random Forest that one of the best model before make hyperparameter optimization"
pre_model3 = XGBRegressor().fit(X, y)
feature_imp3 = pd.DataFrame({'Feature': X.columns, 'Value': pre_model3.feature_importances_})
feature_imp3.sort_values("Value", ascending=False)
#plot_importance(pre_model3, X)

# Feature Selection

feature_imp = feature_imp1.merge(feature_imp2["Value"], left_index=True,right_index=True)
feature_imp = feature_imp.merge(feature_imp3["Value"],left_index=True,right_index=True)

feature_imp = feature_imp.rename(columns = {'Value_x': 'LGBM_Score', 'Value_y': 'RF_Score', 'Value' : 'XGB_Score'}, inplace = False)

feature_imp["Weighted_Score"] = feature_imp["LGBM_Score"] * 0.45 + feature_imp["XGB_Score"] * 0.35 + feature_imp["RF_Score"] * 0.20

feature_imp.sort_values("Weighted_Score",ascending=False)



######################################################
# Automated Hyperparameter Optimization
######################################################

rf_params = {"max_depth": [5, 8, 10],
             "max_features": [7,8, 9],
             "min_samples_split": [17,18,20,22],
             "n_estimators": [125,150,175]}

xgboost_params = {"learning_rate": [0.2, 0.15, 0.05],
                  "max_depth": [7, 8,9],
                  "n_estimators": [275, 300,325],
                  "colsample_bytree": [0.5,0.6,0.7]}

lightgbm_params = {"learning_rate": [0.01, 0.50, 0.05],
                   "n_estimators": [50,100,150,200],
                   "colsample_bytree": [0.2,0.3,0.4,0.5]}

regressors = [("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]


best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model




######################################################
# # Stacking & Ensemble Learning
######################################################

"creating best model"

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])])

voting_reg.fit(X, y)


np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))

################################################
# Pipeline Main Function
################################################
import os

def main():
    df = pd.read_csv("HAFTA_08/Ders Notları/hitters.csv")
    X, y = hitters_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_regressor(best_models, X, y)
    os.chdir("HAFTA_08")
    joblib.dump(voting_clf, "voting_clf_hitters.pkl")
    print("Voting_clf has been created")
    return voting_clf

if __name__ == "__main__":
    main()

