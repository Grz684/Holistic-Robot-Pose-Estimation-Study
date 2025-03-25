import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.utils.urdfpytorch import URDF
from lib.config import (BAXTER_DESCRIPTION_PATH, KUKA_DESCRIPTION_PATH,
                    OWI_DESCRIPTION, OWI_KEYPOINTS_PATH, DOFBOT_DESCRIPTION,
                    PANDA_DESCRIPTION_PATH, PANDA_DESCRIPTION_PATH_VISUAL)

robot = URDF.load(PANDA_DESCRIPTION_PATH)
# print(robot.joints)
for joint in robot.joints:
    print(joint.origin)
