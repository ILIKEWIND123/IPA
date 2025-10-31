# Load necessary libraries
# openxlsx: for reading and writing Excel files
# dplyr: for data manipulation (e.g., %>%, select, filter)
# ridge: for performing ridge regression (if needed later)
library(openxlsx)
library(dplyr)
library(ridge)

# --- Data Loading and Preprocessing ---

# Read training data from the clipboard (e.g., copied from Excel)
data0 <- read.table("clipboard", header = TRUE, sep = '') # read train data
# Read testing data from the clipboard
data2 <- read.table("clipboard", header = TRUE, sep = '') # read test data

# Assign training data to a working variable 'data'
data <- data0

# Data Imputation: Fill missing values (or smooth data) using linear regression predictions
# Loop through each column (starting from the 2nd column, assuming 1st is the target)
for (i in 2:ncol(data)) {
  # Extract target variable (y) and current feature (x)
  y <- data[, 1]
  x <- data[, i]
  
  # Create a dataframe for the linear model
  modeldata <- data.frame(x = x, y = y)
  
  # Fit a linear model (y ~ x) and generate predictions
  pre <- predict(lm(y ~ ., data = modeldata))
  
  # Replace the original column with the predicted values
  data[, i] <- pre
}

# --- Grey Relational Analysis (GRA) ---

# Extract the reference series (first column, e.g., CHL) and evaluation series (all other columns)
ref_series <- data[, 1]
eval_series <- data[, -1]

# Define a function to perform Grey Relational Analysis
grey_relational_analysis <- function(ref, eval_series, rho = 0.5) {
  # ref: reference series (the ideal sequence)
  # eval_series: matrix or dataframe of series to be evaluated
  # rho: distinguishing coefficient, usually set to 0.5 (values between 0 and 1)
  
  # Normalize the data by dividing by the mean of each series
  ref <- ref / mean(ref)
  eval_series <- apply(eval_series, 2, function(col) {
    col / mean(col)
  })
  
  # Calculate the absolute difference matrix between reference and each evaluation series
  diff_matrix <- abs(sweep(eval_series, 1, ref, "-"))
  
  # Calculate the global minimum and maximum differences
  min_diff <- min(diff_matrix)
  max_diff <- max(diff_matrix)
  
  # Calculate the relational coefficients using the grey relational formula
  relational_coefficients <- (min_diff + rho * max_diff) / (diff_matrix + rho * max_diff)
  
  # Calculate the relational grade (mean of coefficients for each series)
  relational_grades <- colMeans(relational_coefficients)
  
  return(relational_grades)
}

# Execute the GRA function and store the results
results <- grey_relational_analysis(ref_series, eval_series)

# --- Process and Rank the GRA Results ---

# Get the band/feature names (all columns except the first one)
bands <- colnames(data0)[-1]

# Convert results to a numeric vector
GRD <- results %>% as.numeric()

# Create a dataframe to associate each band with its Grey Relational Grade (GRD)
AGRD <- data.frame(bands = bands, GRD = GRD)

# Sort the dataframe by GRD in descending order (highest association first)
AGRD <- AGRD[order(AGRD[, 2], decreasing = TRUE), ]

# --- Split Data into Deciles Based on GRD Ranking ---

# Create 10 subsets of the training data (data0) by selecting the target variable ('CHL')
# and bands from different segments of the sorted GRD list (top 10%, next 10%, etc.)
# This effectively creates datasets with bands of decreasing relational importance.

data_0.1 <- data0[append("CHL", AGRD[1:216, 1] %>% as.character())]
data_0.2 <- data0[append("CHL", AGRD[217:431, 1] %>% as.character())]
data_0.3 <- data0[append("CHL", AGRD[432:646, 1] %>% as.character())]
data_0.4 <- data0[append("CHL", AGRD[647:861, 1] %>% as.character())]
data_0.5 <- data0[append("CHL", AGRD[862:1076, 1] %>% as.character())]
data_0.6 <- data0[append("CHL", AGRD[1077:1291, 1] %>% as.character())]
data_0.7 <- data0[append("CHL", AGRD[1292:1506, 1] %>% as.character())]
data_0.8 <- data0[append("CHL", AGRD[1507:1721, 1] %>% as.character())]
data_0.9 <- data0[append("CHL", AGRD[1722:1936, 1] %>% as.character())]
data_1.0 <- data0[append("CHL", AGRD[1937:2151, 1] %>% as.character())]

# --- Export Training Data Subsets to an Excel File ---

# Create a new Excel workbook
wb <- createWorkbook()

# Create a worksheet for each data subset and write the corresponding data
addWorksheet(wb, "data_0.1")
writeData(wb, "data_0.1", data_0.1 %>% as.data.frame())
# ... (repeated for all subsets from data_0.2 to data_1.0)
addWorksheet(wb, "data_0.2")
writeData(wb, "data_0.2", data_0.2 %>% as.data.frame())
addWorksheet(wb, "data_0.3")
writeData(wb, "data_0.3", data_0.3 %>% as.data.frame())
addWorksheet(wb, "data_0.4")
writeData(wb, "data_0.4", data_0.4 %>% as.data.frame())
addWorksheet(wb, "data_0.5")
writeData(wb, "data_0.5", data_0.5 %>% as.data.frame())
addWorksheet(wb, "data_0.6")
writeData(wb, "data_0.6", data_0.6 %>% as.data.frame())
addWorksheet(wb, "data_0.7")
writeData(wb, "data_0.7", data_0.7 %>% as.data.frame())
addWorksheet(wb, "data_0.8")
writeData(wb, "data_0.8", data_0.8 %>% as.data.frame())
addWorksheet(wb, "data_0.9")
writeData(wb, "data_0.9", data_0.9 %>% as.data.frame())
addWorksheet(wb, "data_1.0")
writeData(wb, "data_1.0", data_1.0 %>% as.data.frame())

# Save the workbook to an Excel file named "GRA-TRAIN.xlsx"
saveWorkbook(wb, "GRA-TRAIN.xlsx", overwrite = TRUE) # Fixed typo: overwrite <- T to overwrite = T

############
# --- Repeat the Process for the Test Data ---

# Create the same 10 subsets for the test data (data2) using the GRD ranking derived from the training data
data_0.1 <- data2[append("CHL", AGRD[1:216, 1] %>% as.character())]
data_0.2 <- data2[append("CHL", AGRD[217:431, 1] %>% as.character())]
data_0.3 <- data2[append("CHL", AGRD[432:646, 1] %>% as.character())]
data_0.4 <- data2[append("CHL", AGRD[647:861, 1] %>% as.character())]
data_0.5 <- data2[append("CHL", AGRD[862:1076, 1] %>% as.character())]
data_0.6 <- data2[append("CHL", AGRD[1077:1291, 1] %>% as.character())]
data_0.7 <- data2[append("CHL", AGRD[1292:1506, 1] %>% as.character())]
data_0.8 <- data2[append("CHL", AGRD[1507:1721, 1] %>% as.character())]
data_0.9 <- data2[append("CHL", AGRD[1722:1936, 1] %>% as.character())]
data_1.0 <- data2[append("CHL", AGRD[1937:2151, 1] %>% as.character())]

# --- Export Test Data Subsets to an Excel File ---

# Create a new workbook for the test data
wa <- createWorkbook()

# Create worksheets and write test data subsets (similar to the training data step)
addWorksheet(wa, "data_0.1")
writeData(wa, "data_0.1", data_0.1 %>% as.data.frame())
# ... (repeated for all subsets)
addWorksheet(wa, "data_0.2")
writeData(wa, "data_0.2", data_0.2 %>% as.data.frame())
addWorksheet(wa, "data_0.3")
writeData(wa, "data_0.3", data_0.3 %>% as.data.frame())
addWorksheet(wa, "data_0.4")
writeData(wa, "data_0.4", data_0.4 %>% as.data.frame())
addWorksheet(wa, "data_0.5")
writeData(wa, "data_0.5", data_0.5 %>% as.data.frame())
addWorksheet(wa, "data_0.6")
writeData(wa, "data_0.6", data_0.6 %>% as.data.frame())
addWorksheet(wa, "data_0.7")
writeData(wa, "data_0.7", data_0.7 %>% as.data.frame())
addWorksheet(wa, "data_0.8")
writeData(wa, "data_0.8", data_0.8 %>% as.data.frame())
addWorksheet(wa, "data_0.9")
writeData(wa, "data_0.9", data_0.9 %>% as.data.frame())
addWorksheet(wa, "data_1.0")
writeData(wa, "data_1.0", data_1.0 %>% as.data.frame())

# Save the test data workbook to "GRA-TEST.xlsx"
saveWorkbook(wa, "GRA-TEST.xlsx", overwrite = TRUE)
####