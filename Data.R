library(httr)
library(jsonlite)
library(purrr)
library(tidyverse)

# Function to fetch data for a given page number
fetch_page_data <- function(page_num) {
  url <- paste0("https://backoffice.reforestaction.com/api/projects?fields[0]=externalId&fields[1]=commercialFollowUp&fields[2]=centroidLongitude&fields[3]=centroidLatitude&locale=en&filters[commercialFollowUp][$notNull]=true&filters[centroidLatitude][$notNull]=true&filters[centroidLongitude][$notNull]=true&sort[0]=createdDate%3Adesc&pagination[page]=", page_num, "&pagination[pageSize]=100")
  response <- GET(url)
  if (http_type(response) == "application/json") {
    content(response, as = "text", encoding = "UTF-8") %>%
      fromJSON(flatten = TRUE)
  } else {
    stop("Unexpected response format")
  }
}

# Fetch data for all pages and combine into one dataset
all_data <- lapply(1:11, fetch_page_data)
merged_data <- do.call(rbind, all_data)

# Convert merged data to data frame
merged_df <- as.data.frame(merged_data)

# Print first few rows of the merged dataset
print(head(merged_df))









# Extract the "data" column containing the list of data frames
data_list <- merged_df$data

# Combine all data frames into one
merged_data <- map_dfr(data_list, bind_rows)

# Print first few rows of the merged dataframe
print(head(merged_data))



# Convert data to JSON
json_data <- toJSON(merged_data)

# Write JSON data to a file
writeLines(json_data, "data.json")



#convert to CSV

write.csv2(merged_data, "data.csv")
