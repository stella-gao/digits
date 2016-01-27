# Plot results of neural networks
library(ggplot2)

cnn_2d_results <- data.frame(epoch = rep(1:8, 8),
                            values = c(6.6163, 2.2952, 1.5238, 0.7705, 0.5397,
                                      0.4187, 0.3722, 0.3483,
                                      0.1515, 0.3106, 0.6061, 0.8939, 0.9773,
                                      0.9924, 1.0000, 1.0000,
                                      2.1955, 1.7771, 1.0873, 0.6364, 0.5256,
                                      0.4574, 0.4709, 0.4610,
                                      0.2273, 0.3636, 0.6932, 0.8068, 0.8295,
                                      0.8295, 0.8636, 0.8636),
                            key = as.factor(rep(c("loss", "acc", "val_loss", "val_acc"), each = 8))
                          )

cnn_2d_plot <-  ggplot(data = cnn_2d_results, aes(x = epoch, y = values, group = key, colour = key))
cnn_2d_plot <-  cnn_2d_plot + geom_line() + geom_point() + theme_bw() + 
                scale_color_brewer(palette = "Spectral")
print(cnn_2d_plot)
