import xml.etree.ElementTree as xml_et
import numpy as np
import cv2
import noise
import os
from pathlib import Path


def get_asset_path(robot_name: str = "go1") -> Path:
    current_file = Path(__file__).resolve()
    quadruped_ctrl_dir = current_file.parent.parent  # utils -> quadruped_ctrl
    assets_dir = quadruped_ctrl_dir / "assets" / "robot" / robot_name
    
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")
    
    return assets_dir


# zyx euler angle to quaternion
def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


# zyx euler angle to rotation matrix
def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


# 2d rotate
def rot2d(x, y, yaw):
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny


# 3d rotate
def rot3d(pos, euler):
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    return " ".join(str(s) for s in vec)


class TerrainGenerator:
    """地形生成器 - 为 MuJoCo 场景添加各种地形元素
    
    Args:
        robot_name: 机器人名称 (默认 'go1')
        base_scene: 基础场景文件名 (默认 'scene.xml')
        output_scene: 输出场景文件名 (默认 'scene_terrain.xml')
    """

    def __init__(self, 
                 robot_name: str = "go1", 
                 base_scene: str = "scene.xml",
                 output_scene: str = "scene_terrain.xml") -> None:
        
        self.robot_name = robot_name
        self.asset_dir = get_asset_path(robot_name)
        self.input_path = self.asset_dir / base_scene
        self.output_path = self.asset_dir / output_scene
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Base scene not found: {self.input_path}")
        
        self.scene = xml_et.parse(str(self.input_path))
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")

    # Add Box to scene
    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1, 0.1]):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
    
    def AddGeometry(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1],
               geo_type="box"):
        
        # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddStairs(self,
                  init_pos=[1.0, 0.0, 0.0],
                  yaw=0.0,
                  width=0.2,
                  height=0.10,
                  length=1.5,
                  stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw], [width, length, height])

    def AddSuspendStairs(self,
                         init_pos=[1.0, 0.0, 0.0],
                         yaw=1.0,
                         width=0.2,
                         height=0.15,
                         length=1.5,
                         gap=0.1,
                         stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw],
                        [width, length, abs(height - gap)])

    def AddRoughGround(self,
                       init_pos=[1.0, 0.0, 0.0],
                       euler=[0.0, -0.0, 0.0],
                       nums=[10, 10],
                       box_size=[0.5, 0.5, 0.5],
                       box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2],
                       box_size_rand=[0.05, 0.05, 0.05],
                       box_euler_rand=[0.2, 0.2, 0.2],
                       separation_rand=[0.05, 0.05]):

        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(
            separation_rand) * np.random.uniform(-1.0, 1.0, 2)
        for i in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for j in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(
                    box_size_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(
                    box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(
                    separation_rand) * np.random.uniform(-1.0, 1.0, 2)

                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    def AddPerlinHeighField(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[1.0, 1.0],  # width and length
            height_scale=0.2,  # max height
            negative_height=0.2,  # height in the negative direction of z axis
            image_width=128,  # height field image size
            img_height=128,
            smooth=100.0,  # smooth scale
            perlin_octaves=6,  # perlin noise parameter
            perlin_persistence=0.5,
            perlin_lacunarity=2.0,
            output_hfield_image="height_field.png"):

        # Generating height field based on perlin noise
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(image_width):
            for x in range(image_width):
                # Perlin noise
                noise_value = noise.pnoise2(x / smooth,
                                            y / smooth,
                                            octaves=perlin_octaves,
                                            persistence=perlin_persistence,
                                            lacunarity=perlin_lacunarity)
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

        # 保存到 assets 目录
        output_img_path = self.asset_dir / output_hfield_image
        cv2.imwrite(str(output_img_path), terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = output_hfield_image  # 相对路径

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)


    def AddDownStairs(self,
                      init_pos=[1.0, 0.0, 0.0],
                      yaw=0.0,
                      width=0.2,
                      height=0.10,
                      length=1.5,
                      stair_nums=10):
        
        # 初始局部高度（第一级下楼台阶的高度）
        # 假设 init_pos 是平台的高度，那么第一级台阶应该比平台低一个 height
        for i in range(stair_nums):
            # 计算当前台阶相对于 init_pos 的偏移
            # i=0 时，移动一个 width，高度下降一个 height
            rel_x = (i + 1) * width
            rel_z = - (i + 0.5) * height # -0.5 是为了对齐 box 的中心
            
            x_offset, y_offset = rot2d(rel_x, 0.0, yaw)
            
            self.AddBox([x_offset + init_pos[0], y_offset + init_pos[1], rel_z + init_pos[2]],
                        [0.0, 0.0, yaw], [width, length, height])
            
    def AddHeighFieldFromImage(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[2.0, 1.6],  # width and length
            height_scale=0.02,  # max height
            negative_height=0.1,  # height in the negative direction of z axis
            input_img=None,
            output_hfield_image="height_field.png",
            image_scale=[1.0, 1.0],  # reduce image resolution
            invert_gray=False):

        # 支持绝对路径或相对于 assets 目录的路径
        if not os.path.isabs(input_img):
            input_img = str(self.asset_dir / input_img)
        
        if not os.path.exists(input_img):
            raise FileNotFoundError(f"Input image not found: {input_img}")
        
        input_image = cv2.imread(input_img)
        if input_image is None:
            raise ValueError(f"Failed to load image: {input_img}")

        width = int(input_image.shape[1] * image_scale[0])
        height = int(input_image.shape[0] * image_scale[1])
        resized_image = cv2.resize(input_image, (width, height),
                                   interpolation=cv2.INTER_AREA)
        terrain_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        if invert_gray:
            terrain_image = 255 - terrain_image
        
        # 保存到 assets 目录
        output_img_path = self.asset_dir / output_hfield_image
        cv2.imwrite(str(output_img_path), terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "image_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = output_hfield_image  # 相对路径

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "image_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def Save(self):
        """保存生成的地形场景文件"""
        self.scene.write(str(self.output_path))
        print(f"✅ Terrain scene saved to: {self.output_path}")
        return str(self.output_path)


if __name__ == "__main__":
    tg = TerrainGenerator(robot_name="go1", output_scene="scene_stairs.xml")
    
    # 1. 上楼 (起于 2.0，终于 3.6，最高处 0.8)
    tg.AddStairs(init_pos=[2.0, 0.0, 0.0], yaw=0.0, width=0.2, height=0.1, length=1.5, stair_nums=8)
    tg.AddBox(position=[3.85, 0.0, 0.75], euler=[0,0,0], size=[0.5, 1.5, 0.1])
    tg.AddDownStairs(init_pos=[4.0, 0.0, 0.8], yaw=0.0, width=0.2, height=0.1, length=1.5, stair_nums=8)
    print()
    output_file = tg.Save()
    
