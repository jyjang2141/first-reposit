##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

library(dplyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

movies <- c("Forrest Gump", "Jurassic Park (1993)", "Pulp Fiction", "Shawshank", "Speed 2: Cruise Control")
rating_num <- sapply(movies, function(x) {
  sum(str_count(edx$title, x))
})

#Check for na
sum(is.na(edx))
#Preview edx set
head(edx)

#split data into train set and test set
set.seed(2)
test_ind <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
test_set <- edx[test_ind,]
train_set <- edx[-test_ind,]

#First Model
mu <- mean(train_set$rating)
  #Add RMSE to tibble
rmse_list <- tibble(model = "mu", rmse = RMSE(test_set$rating, mu))
rmse_list

#Second Model - Movie Effects
b_i <- train_set |> group_by(movieId) |> reframe(b_i = mean(rating - mu))
  #Produce pred and rmse
rmse_movie <- left_join(test_set, b_i, by = "movieId") |> 
  mutate(pred = mu + b_i) |> 
  reframe(rmse = RMSE(rating, pred, na.rm = TRUE))
  #Add RMSE to tibble
rmse_list[nrow(rmse_list) + 1,] <- c(model = "mu + movie", rmse = rmse_movie)
rmse_list

#Third Model - User Effects
b_u <- left_join(train_set, b_i, by = "movieId") |> 
  group_by(userId) |> 
  reframe(b_u = mean(rating - mu - b_i))
  #Produce pred and rmse
rmse_user <- left_join(test_set, b_u, by = "movieId") |> 
  left_join(fit_users, by = "userId") |> 
  mutate(pred = mu + b_i + b_u) |> 
  reframe(rmse = RMSE(rating, pred, na.rm = TRUE))
  #Add RMSE to tibble
rmse_list[nrow(rmse_list) + 1,] <- c(model = "mu + movie + user", rmse = rmse_user)
rmse_list

#Fourth Model - Regularized Users and Movies
  #Plot showing number of movies rated and ratings
movie_number_plot <- edx |> group_by(movieId) |> 
  reframe(ratings = mean(rating), title = title, n=n()) |> 
  arrange(desc(ratings)) |>
  ggplot(aes(n, ratings)) +
  geom_point() +
  ggtitle("Number of Movies Rated vs Ratings") +
  xlab("Number of Movies Rated") + 
  ylab("Rating")

  #Plot showing number of users and ratings
user_number_plot <- edx |> group_by(userId) |> 
  reframe(ratings = mean(rating), title = title, n=n()) |> 
  arrange(desc(ratings)) |>
  ggplot(aes(n, ratings)) +
  geom_point() + 
  ggtitle("Number of Users vs Ratings") +
  xlab("Number of Users") + 
  ylab("Rating")

  #Cross validation to choose lambda
lambdas <- seq(0, 10, 0.1)
rmses <- sapply(lambdas, function(lambda) {
  b_i <- train_set |>
    group_by(movieId) |>
    reframe(b_i = sum(rating - mu) / (n() + lambda))
  b_u <- left_join(train_set, b_i, by = "movieId") |>
    group_by(userId) |>
    reframe(b_u = sum(rating - mu - b_i) / (n() + lambda))
  left_join(test_set, b_i, by = "movieId") |> 
    left_join(b_u, by = "userId") |>
    mutate(pred = mu + b_i + b_u) |> 
    reframe(rmse = RMSE(rating, pred, na.rm = TRUE)) |>
    pull(rmse)
})

  #Visualize which lambda minimizes RMSE
qplot(lambdas, rmses, geom = "line")
lambda <- lambdas[which.min(rmses)]

  #Produce pred and rmse
b_i_reg <- train_set |>
  group_by(movieId) |>
  reframe(b_i_reg = sum(rating - mu) / (n() + lambda))
b_u_reg <- left_join(train_set, b_i_reg, by = "movieId") |>
  group_by(userId) |>
  reframe(b_u_reg = sum(rating - mu - b_i_reg) / (n() + lambda))
rmse_reg <- left_join(test_set, b_i_reg, by = "movieId") |> 
  left_join(b_u_reg, by = "userId") |>
  mutate(pred = mu + b_i_reg + b_u_reg) |> 
  reframe(rmse = RMSE(rating, pred, na.rm = TRUE)) |>
  pull(rmse)
  #Add RMSE to tibble
rmse_list[nrow(rmse_list) + 1,] <- list(model = "mu + movie_reg + user_reg", rmse = rmse_reg)
rmse_list

#Fifth Model - Genre Effect
  #Table showing genres and respective ratings
genre_rating_table <- train_set |> group_by(genres) |> 
  reframe(rating = mean(rating)) |>
  arrange(desc(rating)) 
  #Plot showing number of genres and ratings
train_set |> group_by(genres) |> 
  reframe(rating = mean(rating), n = n()) |>
  ggplot(aes(n, rating)) +
  geom_point() +
  ggtitle("Number of Genres vs Ratings") +
  xlab("Number of Genres") + 
  ylab("Rating")

  #Cross validation to choose lambda
lambdas <- seq(0, 15, 1)
rmses <- sapply(lambdas, function(lambda) {
  b_i <- train_set |>
    group_by(movieId) |>
    reframe(b_i = sum(rating - mu) / (n() + lambda))
  b_u <- left_join(train_set, b_i, by = "movieId") |>
    group_by(userId) |>
    reframe(b_u = sum(rating - mu - b_i) / (n() + lambda))
  b_g <- left_join(train_set, b_i, by = "movieId") |>
    left_join(b_u, by = "userId") |>
    group_by(genres) |>
    reframe(b_g = sum(rating - mu - b_i - b_u) / (n() + lambda))
  
  left_join(test_set, b_i, by = "movieId") |> 
    left_join(b_u, by = "userId") |>
    left_join(b_g, by = "genres") |>
    mutate(pred = mu + b_i + b_u + b_g) |> 
    reframe(rmse = RMSE(rating, pred, na.rm = TRUE)) |>
    pull(rmse)
})

  #Visualize which lambda minimizes RMSE
qplot(lambdas, rmses, geom = "line")
lambda <- lambdas[which.min(rmses)]

  #Produce pred and rmse
b_i_reg <- train_set |>
  group_by(movieId) |>
  reframe(b_i_reg = sum(rating - mu) / (n() + lambda))
b_u_reg <- left_join(train_set, b_i_reg, by = "movieId") |>
  group_by(userId) |>
  reframe(b_u_reg = sum(rating - mu - b_i_reg) / (n() + lambda))
b_g_reg <- left_join(train_set, b_i_reg, by = "movieId") |>
  left_join(b_u_reg, by = "userId") |>
  group_by(genres) |>
  reframe(b_g_reg = sum(rating - mu - b_i_reg - b_u_reg) / (n() + lambda))
rmse_reg_genre <- left_join(test_set, b_i_reg, by = "movieId") |> 
  left_join(b_u_reg, by = "userId") |>
  left_join(b_g_reg, by = "genres") |>
  mutate(pred = mu + b_i_reg + b_u_reg + b_g_reg) |> 
  reframe(rmse = RMSE(rating, pred, na.rm = TRUE)) |>
  pull(rmse)
  #Add RMSE to tibble
rmse_list[nrow(rmse_list) + 1,] <- list(model = "mu + movie_reg + user_reg + genre_reg", rmse = rmse_reg_genre)
rmse_list

#Sixth Model - Year Effect
  #Add years (year the movie was released) column
train_set <- train_set |> mutate(years = as.numeric(str_sub(title, start= -5, end = -2)))
test_set <- test_set |> mutate(years = as.numeric(str_sub(title, start= -5, end = -2)))

  #Plot showing years released and ratings
train_set |> group_by(years) |> 
  reframe(rating = mean(rating), years = years) |>
  ggplot(aes(years, rating)) +
  geom_smooth() +
  ggtitle("Years Released vs Rating") +
  xlab("Years Released") + 
  ylab("Rating")

  #Prepare data for years plot
year_number_data <- train_set |> group_by(years) |> 
  reframe(n = n())

  #Plot showing years released and number of ratings
year_number_data[-which.max(year_number_data$n),] |>
  ggplot(aes(years, n)) +
  geom_point() +
  ggtitle("Years Released vs Number of Ratings") +
  xlab("Years Released") + 
  ylab("Number of Ratings")

  #Cross validation to choose lambda
lambdas <- seq(0, 10, 1)
rmses <- sapply(lambdas, function(lambda) {
  b_i <- train_set |>
    group_by(movieId) |>
    reframe(b_i = sum(rating - mu) / (n() + lambda))
  b_u <- left_join(train_set, b_i, by = "movieId") |>
    group_by(userId) |>
    reframe(b_u = sum(rating - mu - b_i) / (n() + lambda))
  b_g <- left_join(train_set, b_i, by = "movieId") |>
    left_join(b_u, by = "userId") |>
    group_by(genres) |>
    reframe(b_g = sum(rating - mu - b_i - b_u) / (n() + lambda))
  b_y <- left_join(train_set, b_i, by = "movieId") |>
    left_join(b_u, by = "userId") |>
    left_join(b_g, by = "genres") |>
    group_by(years) |>
    reframe(b_y = sum(rating - mu - b_i - b_u - b_g) / (n() + lambda))
  
  left_join(test_set, b_i, by = "movieId") |> 
    left_join(b_u, by = "userId") |>
    left_join(b_g, by = "genres") |>
    left_join(b_y, by = "years") |>
    mutate(pred = mu + b_i + b_u + b_g + b_y) |> 
    reframe(rmse = RMSE(rating, pred, na.rm = TRUE)) |>
    pull(rmse)
})

  #Visualize which lambda minimizes RMSE
qplot(lambdas, rmses, geom = "line")
lambda <- lambdas[which.min(rmses)]
  
  #Produce pred and rmse
b_i_reg <- train_set |>
  group_by(movieId) |>
  reframe(b_i_reg = sum(rating - mu) / (n() + lambda))
b_u_reg <- left_join(train_set, b_i_reg, by = "movieId") |>
  group_by(userId) |>
  reframe(b_u_reg = sum(rating - mu - b_i_reg) / (n() + lambda))
b_g_reg <- left_join(train_set, b_i_reg, by = "movieId") |>
  left_join(b_u_reg, by = "userId") |>
  group_by(genres) |>
  reframe(b_g_reg = sum(rating - mu - b_i_reg - b_u_reg) / (n() + lambda))
b_y_reg <- left_join(train_set, b_i_reg, by = "movieId") |>
  left_join(b_u_reg, by = "userId") |>
  left_join(b_g_reg, by = "genres") |> 
  group_by(years) |>
  reframe(b_y_reg = sum(rating - mu - b_i_reg - b_u_reg) / (n() + 14))

rmse_reg_years <- left_join(test_set, b_i_reg, by = "movieId") |> 
  left_join(b_u_reg, by = "userId") |>
  left_join(b_g_reg, by = "genres") |>
  left_join(b_y_reg, by = "years") |>
  mutate(pred = mu + b_i_reg + b_u_reg + b_g_reg + b_y_reg) |> 
  reframe(rmse = RMSE(rating, pred, na.rm = TRUE)) |>
  pull(rmse)
  #Add RMSE to tibble
rmse_list[nrow(rmse_list) + 1,] <- list(model = "mu + movie_reg + user_reg + genre_reg + years_reg", rmse = rmse_reg_years)
rmse_list

#Final Holdout Test
  #Add years column
edx <- edx |> mutate(years = as.numeric(str_sub(title, start= -5, end = -2)))
final_holdout_test <- final_holdout_test |> mutate(years = as.numeric(str_sub(title, start= -5, end = -2)))

  #Cross validation to choose lambda
lambdas <- seq(0, 15, 1)
rmses <- sapply(lambdas, function(lambda) {
  b_i <- edx |>
    group_by(movieId) |>
    reframe(b_i = sum(rating - mu) / (n() + lambda))
  b_u <- left_join(edx, b_i, by = "movieId") |>
    group_by(userId) |>
    reframe(b_u = sum(rating - mu - b_i) / (n() + lambda))
  b_g <- left_join(edx, b_i, by = "movieId") |>
    left_join(b_u, by = "userId") |>
    group_by(genres) |>
    reframe(b_g = sum(rating - mu - b_i - b_u) / (n() + lambda))
  b_y <- left_join(edx, b_i, by = "movieId") |>
    left_join(b_u, by = "userId") |>
    left_join(b_g, by = "genres") |>
    group_by(years) |>
    reframe(b_y = sum(rating - mu - b_i - b_u - b_g) / (n() + lambda))
  
  left_join(final_holdout_test, b_i, by = "movieId") |> 
    left_join(b_u, by = "userId") |>
    left_join(b_g, by = "genres") |>
    left_join(b_y, by = "years") |>
    mutate(pred = mu + b_i + b_u + b_g + b_y) |> 
    reframe(rmse = RMSE(rating, pred, na.rm = TRUE)) |>
    pull(rmse)
})

  #Visualize which lambda minimizes RMSE
qplot(lambdas, rmses, geom = "line")
lambda <- lambdas[which.min(rmses)]

  #Produce preds and rmse
b_i_reg <- edx |>
  group_by(movieId) |>
  reframe(b_i_reg = sum(rating - mu) / (n() + lambda))
b_u_reg <- left_join(edx, b_i_reg, by = "movieId") |>
  group_by(userId) |>
  reframe(b_u_reg = sum(rating - mu - b_i_reg) / (n() + lambda))
b_g_reg <- left_join(edx, b_i_reg, by = "movieId") |>
  left_join(b_u_reg, by = "userId") |>
  group_by(genres) |>
  reframe(b_g_reg = sum(rating - mu - b_i_reg - b_u_reg) / (n() + lambda))
b_y_reg <- left_join(edx, b_i_reg, by = "movieId") |>
  left_join(b_u_reg, by = "userId") |>
  left_join(b_g_reg, by = "genres") |> 
  group_by(years) |>
  reframe(b_y_reg = sum(rating - mu - b_i_reg - b_u_reg) / (n() + 14))

final_model <- left_join(final_holdout_test, b_i_reg, by = "movieId") |> 
  left_join(b_u_reg, by = "userId") |>
  left_join(b_g_reg, by = "genres") |>
  left_join(b_y_reg, by = "years") |>
  mutate(pred = mu + b_i_reg + b_u_reg + b_g_reg + b_y_reg) |> 
  reframe(rmse = RMSE(rating, pred, na.rm = TRUE)) |>
  pull(rmse)
  #Add RMSE to tibble
rmse_list[nrow(rmse_list) + 1,] <- list(model = "final hold out test", rmse = final_model)
rmse_list
