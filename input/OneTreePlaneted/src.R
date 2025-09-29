# Install and load necessary packages
install.packages("httr")
install.packages("jsonlite")
install.packages("tidyverse")

library(httr)
library(jsonlite)
library(tidyverse)

# Base URL
base_url <- 'https://services8.arcgis.com/pKd0yHtOd6niWQzj/arcgis/rest/services/AllProjects_unrefined_Fri13/FeatureServer/146/query'

# Parameters
params <- list(
  f = 'json',
  returnIdsOnly = 'false',
  returnCountOnly = 'false',
  orderByFields = '',
  outSR = '102100',
  returnGeometry = 'true',
  spatialRel = 'esriSpatialRelIntersects',
  where = "Project_Status IN ('Complete','Planted/Reporting','OTP Reporting','Monitoring & Maintenance')"
)

# Sending GET request
response <- GET(url = base_url, query = params)

# Check response
if (status_code(response) == 200) {
  data <- content(response, as = "parsed", type = "application/json")
  
  # Extract features
  features <- data$features
  
  # Extract attributes and geometry
  extract_features <- function(feature) {
    if (!is.null(feature$attributes) && !is.null(feature$geometry)) {
      attributes <- feature$attributes
      geometry <- feature$geometry
      combined <- data.frame(
        Project_Title = ifelse(is.null(attributes$Project_Title), NA, attributes$Project_Title),
        Longitude = ifelse(is.null(geometry$x), NA, geometry$x),
        Latitude = ifelse(is.null(geometry$y), NA, geometry$y),
        stringsAsFactors = FALSE
      )
      return(combined)
    } else {
      return(data.frame(
        Project_Title = NA,
        Longitude = NA,
        Latitude = NA,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Apply the extraction function to all features and combine the results into a dataframe
  extracted_data <- do.call(rbind, lapply(features, extract_features))
  
  # Print the cleaned dataframe
  print(extracted_data)
} else {
  print(paste("Failed to retrieve data:", status_code(response)))
}


# Convert dataframe to JSON
json_data <- toJSON(extracted_data, pretty = TRUE)

# Write JSON data to a file
write(json_data, file = "OneTreePlanted.json")

