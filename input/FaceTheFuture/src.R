

# Install required packages if not already installed
install.packages("httr")
install.packages("jsonlite")
install.packages("dplyr")

# Load required libraries
library(httr)
library(jsonlite)
library(dplyr)

# Define the API URLs
api_urls <- c(
  "https://services3.arcgis.com/Uv1nDNsp84KcmYfH/arcgis/rest/services/Klimaatbossen_NL_PV/FeatureServer/1/query?f=geojson&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=FID&outFields=FID&resultType=tile&spatialRel=esriSpatialRelIntersects&geometryType=esriGeometryEnvelope&defaultSR=102100",
  "https://services3.arcgis.com/Uv1nDNsp84KcmYfH/arcgis/rest/services/Kibale_ReforestationProject_PV/FeatureServer/0/query?f=geojson&cacheHint=true&maxRecordCountFactor=100&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=*&resultType=tile&returnGeometry=true&spatialRel=esriSpatialRelIntersects&geometryType=esriGeometryEnvelope&defaultSR=102100"
)

# Function to fetch data from API and convert to data frame
fetch_data <- function(url) {
  response <- httr::GET(url)
  data <- httr::content(response, as = "text")
  json_data <- jsonlite::fromJSON(data)
  features <- json_data$features
  df <- jsonlite::fromJSON(jsonlite::toJSON(features))
  return(df)
}

# Fetch data from both APIs
data_frames <- lapply(api_urls, fetch_data)

# Combine data frames
combined_df <- bind_rows(data_frames)

# Display the combined data frame
print(combined_df)


# Save combined data frame as CSV
write.csv2(combined_df, "reforestation_projects_data.csv", row.names = FALSE)

# Save combined data frame as JSON
write_json(combined_df, "reforestation_projects_data.json")
