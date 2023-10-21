from pprint import pprint
from collections import deque
import copy
import numpy as np
from PIL import Image
import numpy as np
import os
import json

START_COL = "S"
END_COL = "E"
VISITED_COL = "x"
OBSTACLE_COL = "#"
PATH_COL = "@"



def douglas_peucker(points, epsilon):
    """
    Apply Douglas-Peucker algorithm to reduce the number of points in the path.

    :param points: List of points representing the path.
    :param epsilon: Threshold distance for point reduction.
    :return: Reduced list of points.
    """
    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(points) - 1

    for i in range(1, end):
        d = point_line_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        rec_results1 = douglas_peucker(points[:index + 1], epsilon)
        rec_results2 = douglas_peucker(points[index:], epsilon)

        # Build the result list
        result_list = rec_results1[:-1] + rec_results2
    else:
        result_list = [points[0], points[end]]

    return result_list


def point_line_distance(point, start, end):
    """
    Calculate the perpendicular distance from a point to a line.

    :param point: Point (x, y) to check.
    :param start: Start point (x, y) of the line segment.
    :param end: End point (x, y) of the line segment.
    :return: Perpendicular distance from the point to the line.
    """
    px, py = point
    sx, sy = start
    ex, ey = end

    line_length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
    if line_length == 0:
        return ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
    u = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / (line_length ** 2)
    closest_x = sx + u * (ex - sx)
    closest_y = sy + u * (ey - sy)
    distance = ((closest_x - px) ** 2 + (closest_y - py) ** 2) ** 0.5

    return distance


def draw_path_with_douglas_peucker(path, grid):
    # Apply the Douglas-Peucker algorithm to simplify the path
    reduced_path = douglas_peucker(path, epsilon=1)


    # Connect the points in the reduced path with lines
    for i in range(len(reduced_path) - 1):
        start = reduced_path[i]
        end = reduced_path[i + 1]
        line = get_line_points(start, end)
        for (x, y) in line:
            grid[x][y] = PATH_COL

    # Draw start and end points
    start_pos = reduced_path[0]
    end_pos = reduced_path[-1]
    grid[start_pos[0]][start_pos[1]] = START_COL
    grid[end_pos[0]][end_pos[1]] = END_COL

    return grid

def get_line_points(start, end):
    """Bresenham's Line Algorithm to get points of the line between start and end."""
    x1, y1 = start
    x2, y2 = end
    points = []
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    y = y1
    y_step = None
    if y1 < y2:
        y_step = 1
    else:
        y_step = -1
    for x in range(x1, x2 + 1):
        if is_steep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:
        points.reverse()
    return points
def thicken_collision_points(pixel_data, radius=2):
    # Make a deep copy of the input data to avoid modifying it directly
    thickened_data = copy.deepcopy(pixel_data)

    height = len(pixel_data)
    width = len(pixel_data[0]) if height > 0 else 0

    modified_points = 0  # Counter to see how many points are modified

    for i in range(height):
        for j in range(width):
            if pixel_data[i][j] == '#':  # Checking for collision character
                # Loop through the neighborhood defined by the radius
                for x in range(-radius, radius + 1):
                    for y in range(-radius, radius + 1):
                        new_i = i + x
                        new_j = j + y

                        # Check if the new_i and new_j indices are valid
                        if 0 <= new_i < height and 0 <= new_j < width:
                            if thickened_data[new_i][new_j] != '#':  # Check if the point hasn't been modified before
                                thickened_data[new_i][new_j] = '#'  # Modify the pixel to represent collision
                                modified_points += 1


    return thickened_data
def expand_pixel_data(pixel_data):
    expanded = []
    for row in pixel_data:
        expanded_row = []
        for pixel in row:
            expanded_row.extend([pixel] * 3)
        for _ in range(3):
            expanded.append(expanded_row)
    return expanded

def scan_grid(grid, start=(0, 0)):
    """Scan all grid, so we can find a path from 'start' to any point"""

    q = deque()
    q.append(start)
    came_from = {start: None}
    while len(q) > 0:
        current_pos = q.popleft()
        neighbors = get_neighbors(grid, current_pos[0], current_pos[1])
        for neighbor in neighbors:
            if neighbor not in came_from:
                q.append(neighbor)
                came_from[neighbor] = current_pos

    return came_from

def get_neighbors(grid, row, col):
    height = len(grid)
    width = len(grid[0])

    neighbors = [(row + 1, col), (row, col - 1), (row - 1, col), (row, col + 1)]

    # make path nicer
    if (row + col) % 2 == 0:
        neighbors.reverse()

    # check borders
    neighbors = filter(lambda t: (0 <= t[0] < height and 0 <= t[1] < width), neighbors)
    # check obstacles
    neighbors = filter(lambda t: (grid[t[0]][t[1]] != OBSTACLE_COL), neighbors)

    return neighbors

def parse_pgm(data):
    lines = data.split("\n")
    metadata = {}
    pixel_data = []

    # Loop through lines and parse data
    for line in lines:
        # Skip comments
        if line.startswith("#"):
            continue
        # Check for magic number P2
        elif line == "P2":
            metadata["type"] = "P2"
        # Check for width and height
        elif "width" not in metadata:
            metadata["width"], metadata["height"] = map(int, line.split())
        # Check for max gray value
        elif "max_gray" not in metadata:
            metadata["max_gray"] = int(line)
        # Parse pixel data
        else:
            row = list(map(int, line.split()))
            pixel_data.append(row)

            # Debugging
            if len(row) not in [metadata["width"], 0]:  # 0 is for potential empty lines
                print(f"Unexpected row length {len(row)} at line: {line}")

    # Further debugging to display inconsistent row lengths
    row_lengths = [len(row) for row in pixel_data]
    if len(set(row_lengths)) > 1:
        print("Inconsistent row lengths detected:", set(row_lengths))
        for i, length in enumerate(row_lengths):
            if length != metadata["width"] and length != 0:
                print(f"Row {i} has length {length}")

    return metadata, pixel_data

def replace_values_in_array(pixel_data):
    for i in range(len(pixel_data)):
        for j in range(len(pixel_data[i])):
            if pixel_data[i][j] == 255:
                pixel_data[i][j] = '.'
            elif pixel_data[i][j] == 0:
                pixel_data[i][j] = '#'
    return pixel_data

def write_2d_array_to_file(pixel_data, filename):
    max_width = max(len(str(item)) for row in pixel_data for item in row)  # Find the maximum width of the items
    with open(filename, 'w') as file:
        for row in pixel_data:
            # Create a formatted string with even spacing, write it to the file
            line = ''.join(f'{item:>{max_width+1}}' for item in row)
            file.write(line + '\n')

def write_pgm(pixel_data, filename, max_value=255):
    # Ensure max_value is valid
    max_value = min(max(max_value, 0), 255)

    # Determine the dimensions of the image
    height = len(pixel_data)
    width = len(pixel_data[0]) if height > 0 else 0

    # Write header and pixel data to file
    with open(filename, 'w') as f:
        f.write(f"P2\n{width} {height}\n{max_value}\n")
        for row in pixel_data:
            f.write(' '.join(map(str, row)) + '\n')


def convert_to_numeric(pixel_data):
    """
    Convert a 2D array of symbols to a 2D array of numerical values.

    Symbols and their corresponding values:
        '.' -> 255
        '#' -> 0
        'S' -> 150
        'E' -> 150
        'x' -> 150
        '@' -> 150

    :param pixel_data: 2D array containing symbols.
    :return: A new 2D array with numerical values.
    """
    conversion_dict = {
        '.': 255,
        '#': 0,
        'S': 150,
        'E': 150,
        'x': 150,
        '@': 150
    }

    return [[conversion_dict.get(pixel, 0) for pixel in row] for row in pixel_data]
def find_path(start, end, came_from):
    """Find the shortest path from start to end point"""

    path = [end]

    current = end
    while current != start:
        current = came_from[current]
        path.append(current)

    # reverse to have Start -> Target
    # just looks nicer
    path.reverse()

    return path

def draw_path(path, grid):
    for row, col in path:
        grid[row][col] = PATH_COL

    # draw start and end
    start_pos = path[0]
    end_pos = path[-1]
    grid[start_pos[0]][start_pos[1]] = START_COL
    grid[end_pos[0]][end_pos[1]] = END_COL

    return grid


def init():
    layers = []
    directory_path = os.path.join('/home', 'lrs-ubuntu', 'workspace', 'src', 'template_drone_control', 'src',
                                  'FEI_LRS_2D')
    files_in_directory = os.listdir(directory_path)

    pgm_files = [f for f in files_in_directory if f.endswith('.pgm')]
    index = 0
    for pgm_file in pgm_files:
        file_path = os.path.join(directory_path, pgm_file)

        with open(file_path, "rb") as file:
            byte_data = file.read()
            data = byte_data.decode("utf-8")

        metadata, pixel_data = parse_pgm(data)
        pixel_data = [row for row in pixel_data if len(row) > 0]
        # 1. Replace values
        pixel_data = replace_values_in_array(pixel_data)
        row_lengths = [len(row) for row in pixel_data]
        if len(set(row_lengths)) > 1:
            print("Still have inconsistent row lengths:", set(row_lengths))

        # 2. Expand the pixel data
        pixel_data = expand_pixel_data(pixel_data)

        # 3. Thicken collision points
        pixel_data = thicken_collision_points(pixel_data)

        # Now proceed with the rest
        # Ensure all rows are of consistent length (using the maximum length found)
        # Remove completely empty rows
        pixel_data = [row for row in pixel_data if any(val != 255 for val in row)]

        # Ensure all rows are of consistent length (using the maximum length found)
        max_length = max(len(row) for row in pixel_data)
        filtered_data = [row + [255] * (max_length - len(row)) for row in pixel_data]

        filtered_data_pgm = convert_to_numeric(filtered_data)

        write_pgm(filtered_data_pgm, 'map.pgm')

        start_pos = (750, 900)  # Adjusted for 3x scaling
        directions = scan_grid(filtered_data, start_pos)

        path1 = find_path(start_pos, (150, 105), directions)  # Adjusted end point for 3x scaling

        grid_with_path1 = draw_path_with_douglas_peucker(path1, copy.deepcopy(filtered_data))

        grid_with_path1_converted = convert_to_numeric(grid_with_path1)
        layers.append(grid_with_path1_converted)

        print(path1)
        write_pgm(grid_with_path1_converted, f'path_output-{index}.pgm')
        index += 1

    print('layers num:')
    print(len(layers))
    return layers

if __name__ == "__main__":
    layers = init()
    with open('layers_output.json', 'w') as file:
        json.dump(layers, file)
#'/home/lrs-ubuntu/workspace/src/template_drone_control/src/FEI_LRS_2D/'