# Install httr and jsonlite packages if not already installed
if (!requireNamespace("httr", quietly = TRUE)) {
  install.packages("httr")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}

# Load the libraries
library(httr)
library(jsonlite)



# Define the API endpoint
url <- "https://tree-nation.com/api/projects/"

# Make the GET request to the API
response1 <- GET(url)

# Check if the request was successful
if (http_status(response1)$category == "Success") {
  # Parse the JSON response
  json_data1 <- content(response1, "text", encoding = "UTF-8")
  
  # Convert JSON to dataframe
  df1 <- fromJSON(json_data1, flatten = TRUE)
  
  # View the dataframe
  print(df1)
} else {
  cat("Error: Unable to retrieve data from the API")
}


write.csv(df1, "Projects.csv")