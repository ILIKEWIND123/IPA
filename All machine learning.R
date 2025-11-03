# Load required libraries
library(ggplot2)
library(ForecastComb)
library(EBMAforecast)
library(Boruta)
library(e1071)
library(glmnet)
library(pls)
library(dplyr)
library(xgboost)
library(openxlsx)
library(randomForest)
library(kernlab)
library(caret)
library(h2o)

# Initialize H2O cluster
h2o.init()

# Define the proportion thresholds for different feature subsets
por = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Main loop for each proportion threshold
for (jkl in 1:10) {
  
  # Load training and testing data for current proportion
  Traindata = read.xlsx("D:/TRAINdata.xlsx", sheet = jkl) %>% as.data.frame()
  Testdata = read.xlsx("D:/TESTdata.xlsx", sheet = jkl) %>% as.data.frame()
  
  ##################################################################
  # Train individual base models on full training data
  
  ctrl <- trainControl(
    method = "cv",        # 交叉验证
    number = 5
  )
  
  # 1. Random Forest
  trainpre_model1 = train(CHL ~ ., data = Traindata, method = "rf",trControl = ctrl)
  
  # 2. K-Nearest Neighbors
  trainpre_model2 = train(CHL ~ ., data = Traindata, method = "knn",trControl = ctrl)
  
  # 3. Support Vector Machine with Radial Basis Function Kernel
  trainpre_model3 = train(CHL ~ ., data = Traindata, method = "svmRadial",trControl = ctrl)
  
  # 4. Partial Least Squares Regression
  trainpre_model4 = plsr(CHL ~ ., data = Traindata, ncomp = 10, validation = "CV")
  
  # 5. Lasso and Elastic-Net Regularized Generalized Linear Models
  trainpre_model5 = cv.glmnet(x = Traindata[, -1] %>% as.matrix(), y = Traindata[, 1], nfolds = 5)
  
  # 6. Extreme Gradient Boosting
  trainpre_model6 = train(CHL ~ ., data = Traindata, method = "xgbTree",trControl = ctrl)
  
  # 7. Deep Neural Network using H2O
  h2o.Traindata = as.h2o(Traindata)
  h2o.Testdata = as.h2o(Testdata)
  trainpre_model7 = h2o.deeplearning(
    x = 2:ncol(Traindata), 
    y = 1, 
    training_frame = h2o.Traindata, 
    hidden = c(64, 128, 256, 512, 512, 1024),
    epochs = 200
  )
  
  # 8. Gaussian Process Regression
  trainpre_model8 = train(CHL ~ ., data = Traindata, method = "gaussprRadial",trControl = ctrl)
  
  # Generate predictions on test data using all base models
  test_one_1 = predict(trainpre_model1, newdata = Testdata)
  test_one_2 = predict(trainpre_model2, newdata = Testdata)
  test_one_3 = predict(trainpre_model3, newdata = Testdata)
  test_one_4 = predict(trainpre_model4, newdata = Testdata, 
                       ncomp = ifelse(selectNcomp(trainpre_model4) == 0, 1, selectNcomp(trainpre_model4)))
  test_one_5 = predict(trainpre_model5, newx = Testdata[, -1] %>% as.matrix())
  test_one_6 = predict(trainpre_model6, newdata = Testdata)
  test_one_7 = h2o.predict(trainpre_model7, newdata = h2o.Testdata) %>% as.data.frame()
  test_one_8 = predict(trainpre_model8, newdata = Testdata)
  
  # Create datasets for ensemble forecasting
  # ALL: Includes tree-based models (RF, XGB)
  secdata_test_ALL = data.frame(
    CHL = Testdata$CHL, 
    RF = test_one_1, 
    XGB = test_one_6, 
    KNN = test_one_2, 
    SVM = test_one_3, 
    PLSR = test_one_4 %>% as.numeric(),
    GLM = test_one_5[, 1], 
    DNN = test_one_7[, 1], 
    GP = test_one_8
  )
  
  # NOTREE: Excludes tree-based models (RF, XGB)
  secdata_test_NOTREE = data.frame(
    CHL = Testdata$CHL, 
    KNN = test_one_2, 
    SVM = test_one_3, 
    PLSR = test_one_4 %>% as.numeric(),
    GLM = test_one_5[, 1], 
    DNN = test_one_7[, 1], 
    GP = test_one_8
  )
  ##################################################################
  
  # 5-fold cross-validation for ensemble model calibration
  folds = caret::createFolds(y = Traindata[, 1], k = 5)
  
  OSPM_ALL = NULL    # Out-of-sample predictions for ALL models
  OSPM_NOTREE = NULL # Out-of-sample predictions for NOTREE models
  
  # Cross-validation loop
  for(i in 1:5) {
    
    # Split data into training and validation folds
    fold_test = Traindata[folds[[i]], ]
    fold_train = Traindata[-folds[[i]], ]
    
    # Train base models on current training fold
    CV_trainpre_model1 = train(CHL ~ ., data = fold_train, method = "rf",trControl = ctrl)
    CV_trainpre_model2 = train(CHL ~ ., data = fold_train, method = "knn",trControl = ctrl)
    CV_trainpre_model3 = train(CHL ~ ., data = fold_train, method = "svmRadial",trControl = ctrl)
    CV_trainpre_model4 = plsr(CHL ~ ., data = fold_train, ncomp = 10, validation = "CV")
    CV_trainpre_model5 = cv.glmnet(x = fold_train[, -1] %>% as.matrix(), y = fold_train[, 1], nfolds = 5)
    CV_trainpre_model6 = train(CHL ~ ., data = fold_train, method = "xgbTree",trControl = ctrl)
    
    # H2O Deep Learning
    CV_h2o.Traindata = as.h2o(fold_train)
    CV_h2o.Testdata = as.h2o(fold_test)
    CV_trainpre_model7 = h2o.deeplearning(
      x = 2:ncol(fold_train), 
      y = 1, 
      training_frame = CV_h2o.Traindata, 
      hidden = c(64, 128, 256, 512, 512, 1024),
      epochs = 200
    )
    CV_trainpre_model8 = train(CHL ~ ., data = fold_train, method = "gaussprRadial",trControl = ctrl)
    
    # Generate predictions on validation fold
    CV_test_one_1 = predict(CV_trainpre_model1, newdata = fold_test)
    CV_test_one_2 = predict(CV_trainpre_model2, newdata = fold_test)
    CV_test_one_3 = predict(CV_trainpre_model3, newdata = fold_test)
    CV_test_one_4 = predict(CV_trainpre_model4, newdata = fold_test, 
                            ncomp = ifelse(selectNcomp(CV_trainpre_model4) == 0, 1, selectNcomp(CV_trainpre_model4)))
    CV_test_one_5 = predict(CV_trainpre_model5, newx = fold_test[, -1] %>% as.matrix())
    CV_test_one_6 = predict(CV_trainpre_model6, newdata = fold_test)
    CV_test_one_7 = h2o.predict(CV_trainpre_model7, newdata = CV_h2o.Testdata) %>% as.data.frame()
    CV_test_one_8 = predict(CV_trainpre_model8, newdata = fold_test)
    
    # Combine predictions for ensemble methods
    CV_COM_ALL = data.frame(
      CHL = fold_test$CHL, 
      RF = CV_test_one_1, 
      XGB = CV_test_one_6, 
      KNN = CV_test_one_2, 
      SVM = CV_test_one_3,
      PLSR = CV_test_one_4 %>% as.numeric(), 
      GLM = CV_test_one_5[, 1], 
      DNN = CV_test_one_7[, 1], 
      GP = CV_test_one_8
    )
    
    CV_COM_NOTREE = data.frame(
      CHL = fold_test$CHL, 
      KNN = CV_test_one_2, 
      SVM = CV_test_one_3,
      PLSR = CV_test_one_4 %>% as.numeric(), 
      GLM = CV_test_one_5[, 1], 
      DNN = CV_test_one_7[, 1], 
      GP = CV_test_one_8
    )
    
    # Append to out-of-sample prediction matrices
    OSPM_ALL = rbind(OSPM_ALL, CV_COM_ALL)
    OSPM_NOTREE = rbind(OSPM_NOTREE, CV_COM_NOTREE)
    
    print(paste("Completed fold", i))
  }
  
  ##################################################################
  # Ensemble combination for ALL models (including tree-based)
  
  data_combine_ALL = foreccomb(OSPM_ALL[, 1], OSPM_ALL[, -1] %>% as.matrix(), 
                               secdata_test_ALL[, 1], secdata_test_ALL[, -1] %>% as.matrix())
  
  # Apply different combination methods
  ALL_DF_pre1 = comb_BG(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre2 = comb_CLS(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre3 = comb_EIG1(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre4 = comb_EIG2(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre5 = comb_EIG3(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre6 = comb_EIG4(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre7 = comb_InvW(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre8 = comb_LAD(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre9 = comb_MED(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre10 = comb_NG(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre11 = comb_TA(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre12 = comb_WA(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre13 = comb_OLS(data_combine_ALL)$Forecasts_Test
  ALL_DF_pre14 = comb_SA(data_combine_ALL)$Forecasts_Test
  
  # Bayesian Model Averaging using EBMA
  this.ForecastData = makeForecastData(
    .predCalibration = OSPM_ALL[, -1],
    .outcomeCalibration = OSPM_ALL[, 1],
    .predTest = secdata_test_ALL[, -1]
  )
  
  this.ensemble.em = calibrateEnsemble(this.ForecastData, model = "normal", method = "EM", tol = 1e-4)
  ALL_DF_pre15 = this.ensemble.em@predTest[1:nrow(secdata_test_ALL)]
  
  # Combine all ensemble predictions for ALL models
  cv_combin_ALL = data.frame(
    CHL = Testdata$CHL, 
    ALL_BG = ALL_DF_pre1, 
    ALL_CLG = ALL_DF_pre2, 
    ALL_EIG1 = ALL_DF_pre3, 
    ALL_EIG2 = ALL_DF_pre4, 
    ALL_EIG3 = ALL_DF_pre5, 
    ALL_EIG4 = ALL_DF_pre6,
    ALL_INVW = ALL_DF_pre7, 
    ALL_LAD = ALL_DF_pre8, 
    ALL_MED = ALL_DF_pre9, 
    ALL_NG = ALL_DF_pre10, 
    ALL_TA = ALL_DF_pre11, 
    ALL_WA = ALL_DF_pre12, 
    ALL_OLS = ALL_DF_pre13,
    ALL_SA = ALL_DF_pre14, 
    ALL_BMA = ALL_DF_pre15
  )
  
  # Store combination weights for ALL models
  ALL_weights = data.frame(
    BG = comb_BG(data_combine_ALL)$Weights, 
    CLS = comb_CLS(data_combine_ALL)$Weights, 
    EIG1 = comb_EIG1(data_combine_ALL)$Weights,
    EIG2 = comb_EIG2(data_combine_ALL)$Weights, 
    EIG3 = comb_EIG3(data_combine_ALL)$Weights, 
    EIG4 = comb_EIG4(data_combine_ALL)$Weights,
    INVW = comb_InvW(data_combine_ALL)$Weights, 
    LAD = comb_LAD(data_combine_ALL)$Weights, 
    NG = comb_NG(data_combine_ALL)$Weights,
    OLS = comb_OLS(data_combine_ALL)$Weights, 
    SA = comb_SA(data_combine_ALL)$Weights, 
    BMA = this.ensemble.em@modelWeights
  )
  
  ##################################################################
  # Ensemble combination for NOTREE models (excluding tree-based)
  
  data_combine_NOTREE = foreccomb(OSPM_NOTREE[, 1], OSPM_NOTREE[, -1] %>% as.matrix(), 
                                  secdata_test_NOTREE[, 1], secdata_test_NOTREE[, -1] %>% as.matrix())
  
  # Apply different combination methods for NOTREE models
  NOTREE_DF_pre1 = comb_BG(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre2 = comb_CLS(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre3 = comb_EIG1(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre4 = comb_EIG2(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre5 = comb_EIG3(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre6 = comb_EIG4(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre7 = comb_InvW(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre8 = comb_LAD(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre9 = comb_MED(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre10 = comb_NG(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre11 = comb_TA(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre12 = comb_WA(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre13 = comb_OLS(data_combine_NOTREE)$Forecasts_Test
  NOTREE_DF_pre14 = comb_SA(data_combine_NOTREE)$Forecasts_Test
  
  # Bayesian Model Averaging for NOTREE models
  this.ForecastData = makeForecastData(
    .predCalibration = OSPM_NOTREE[, -1],
    .outcomeCalibration = OSPM_NOTREE[, 1],
    .predTest = secdata_test_NOTREE[, -1]
  )
  
  this.ensemble.em = calibrateEnsemble(this.ForecastData, model = "normal", method = "EM", tol = 1e-4)
  NOTREE_DF_pre15 = this.ensemble.em@predTest[1:nrow(secdata_test_ALL)]
  
  # Combine all ensemble predictions for NOTREE models
  cv_combin_NOTREE = data.frame(
    CHL = Testdata$CHL, 
    NOTREE_BG = NOTREE_DF_pre1, 
    NOTREE_CLG = NOTREE_DF_pre2, 
    NOTREE_EIG1 = NOTREE_DF_pre3, 
    NOTREE_EIG2 = NOTREE_DF_pre4,
    NOTREE_EIG3 = NOTREE_DF_pre5, 
    NOTREE_EIG4 = NOTREE_DF_pre6, 
    NOTREE_INVW = NOTREE_DF_pre7, 
    NOTREE_LAD = NOTREE_DF_pre8, 
    NOTREE_MED = NOTREE_DF_pre9,
    NOTREE_NG = NOTREE_DF_pre10, 
    NOTREE_TA = NOTREE_DF_pre11, 
    NOTREE_WA = NOTREE_DF_pre12, 
    NOTREE_OLS = NOTREE_DF_pre13, 
    NOTREE_SA = NOTREE_DF_pre14,
    NOTREE_BMA = NOTREE_DF_pre15
  )
  
  # Store combination weights for NOTREE models
  NOTREE_weights = data.frame(
    BG = comb_BG(data_combine_NOTREE)$Weights, 
    CLS = comb_CLS(data_combine_NOTREE)$Weights, 
    EIG1 = comb_EIG1(data_combine_NOTREE)$Weights,
    EIG2 = comb_EIG2(data_combine_NOTREE)$Weights, 
    EIG3 = comb_EIG3(data_combine_NOTREE)$Weights, 
    EIG4 = comb_EIG4(data_combine_NOTREE)$Weights,
    INVW = comb_InvW(data_combine_NOTREE)$Weights, 
    LAD = comb_LAD(data_combine_NOTREE)$Weights, 
    NG = comb_NG(data_combine_NOTREE)$Weights,
    OLS = comb_OLS(data_combine_NOTREE)$Weights, 
    SA = comb_SA(data_combine_NOTREE)$Weights, 
    BMA = this.ensemble.em@modelWeights
  )
  
  ##############################
  # Combine all predictions and save results
  
  # Merge base model predictions with ensemble predictions
  cv_combin = cbind(secdata_test_ALL, cv_combin_NOTREE[, -1], cv_combin_ALL[, -1])
  
  # Create Excel workbook and save results
  outall = createWorkbook()
  addWorksheet(outall, "PRE")
  writeData(outall, "PRE", cv_combin %>% as.data.frame())
  
  addWorksheet(outall, "OSPM")
  writeData(outall, "OSPM", OSPM_ALL %>% as.data.frame())
  
  addWorksheet(outall, "ALL_weights")
  writeData(outall, "ALL_weights", ALL_weights %>% as.data.frame())
  
  addWorksheet(outall, "Weight_NOTREE")
  writeData(outall, "Weight_NOTREE", NOTREE_weights %>% as.data.frame())
  
  # Save workbook with proportion-specific filename
  saveWorkbook(outall, paste0("Prediction", por[jkl], ".xlsx"), overwrite = TRUE)
  
  print(paste("Completed proportion", por[jkl]))
}

# Shutdown H2O cluster
h2o.shutdown(prompt = FALSE)