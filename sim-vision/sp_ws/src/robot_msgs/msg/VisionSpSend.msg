# 下位机通过 VISION_SP_SEND_ID 发给上位机的 SP 视觉状态。
uint8 mode
float32[4] q  # w, x, y, z
float32 yaw
float32 yaw_vel
float32 pitch
float32 pitch_vel
float32 bullet_speed
uint16 bullet_count
uint8 camp  # 0: red, 1: blue
