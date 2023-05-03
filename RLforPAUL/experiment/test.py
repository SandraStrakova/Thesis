import numpy as np
import random

dayOfWeek = 7
notification_per_day = 2
max_decsionPerDay = 12
reward = 1.0

max_notification = notification_per_day * dayOfWeek

notification_hours = random.sample(range(8, 20), int(max_notification / dayOfWeek))

print(notification_hours)
