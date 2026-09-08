# 上位机通过 VISION_CTRL_NEW_ID 发给下位机的 SP 视觉控制量。
# mode: 0 = no control, 1 = control, 2 = control and fire
uint8 mode
float32 yaw
float32 yaw_vel
float32 yaw_acc
float32 pitch
float32 pitch_vel
float32 pitch_acc
