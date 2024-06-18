# Load the required library
library(jsonlite)

# Fetch JSON data from the URL
json_data <- httr::GET("https://api.explorer.land/v1/public/spots/polygon?project_slug=ffl-platbos")
json_content <- httr::content(json_data, as = "text")

# Parse JSON string into a list
data_list <- fromJSON(json_content)



results_df <- data_list$features


