# Plot results of neural networks
library(ggplot2)

cnn_2d_results <- data.frame(
  epoch = rep(1:8, 8),
  values = c(6.6163, 2.2952, 1.5238, 0.7705, 0.5397,
            0.4187, 0.3722, 0.3483,
            0.1515, 0.3106, 0.6061, 0.8939, 0.9773,
            0.9924, 1.0000, 1.0000,
            2.1955, 1.7771, 1.0873, 0.6364, 0.5256,
            0.4574, 0.4709, 0.4610,
            0.2273, 0.3636, 0.6932, 0.8068, 0.8295,
            0.8295, 0.8636, 0.8636),
  key = as.factor(rep(c("loss", "acc", "val_loss", 
                        "val_acc"), each = 8))
                          )

# 2D CNN loss plot
plot_df <- subset(cnn_2d_results, key == "loss" | key == "val_loss")
cnn_2d_plot_loss <-  ggplot(data = plot_df, aes(x = epoch, y = values, 
                                                  group = key, colour = key))
cnn_2d_plot_loss <-  cnn_2d_plot_loss + geom_line() + geom_point() + theme_bw() + 
                scale_color_brewer(palette = "Dark2" )
print(cnn_2d_plot_loss)

# 2D CNN acc plot
plot_df <- subset(cnn_2d_results, key == "acc" | key == "val_acc")
cnn_2d_plot_acc <-  ggplot(data = plot_df, aes(x = epoch, y = values, 
                                            group = key, colour = key))
cnn_2d_plot_acc <-  cnn_2d_plot_acc + geom_line() + geom_point() + theme_bw() + 
  scale_color_brewer(palette = "Dark2" )
print(cnn_2d_plot_acc)

cnn_1d_results <- data.frame(
  epoch = rep(1:12, 4),
  values = c( 2.4024, 2.4000, 2.4001, 2.4004,
             2.4000, 2.4000, 2.3999, 2.4001,
             2.4002, 2.3999, 2.4002, 2.3998,
             0.0455, 0.0606, 0.0303, 0.0606,
             0.0606, 0.0530, 0.0455, 0.0379,
             0.0455, 0.0758, 0.0530, 0.0530,
             2.3979, 2.3979, 2.3979, 2.3979,
             2.3979, 2.3979, 2.3979, 2.3979,
             2.3979, 2.3979, 2.3979, 2.3979,
             0.0909, 0.0909, 0.0909, 0.0909,
             0.0909, 0.0909, 0.0909, 0.0909,
             0.0909, 0.0909, 0.0909, 0.0909),
   key = as.factor(rep(c("loss", "acc", "val_loss", 
                         "val_acc"), each = 12))
                              )

# 1D CNN loss plot
plot_df <- subset(cnn_1d_results, key == "loss" | key == "val_loss")
cnn_1d_plot_loss <-  ggplot(data = plot_df, aes(x = epoch, y = values, 
                                                group = key, colour = key))
cnn_1d_plot_loss <-  cnn_1d_plot_loss + geom_line() + geom_point() + theme_bw() + 
  scale_color_brewer(palette = "Dark2" )
print(cnn_1d_plot_loss)

# 1D CNN acc plot
plot_df <- subset(cnn_1d_results, key == "acc" | key == "val_acc")
cnn_2d_plot_acc <-  ggplot(data = plot_df, aes(x = epoch, y = values, 
                                                group = key, colour = key))
cnn_2d_plot_acc <-  cnn_2d_plot_acc + geom_line() + geom_point() + theme_bw() + 
  scale_color_brewer(palette = "Dark2" )
print(cnn_2d_plot_acc)
