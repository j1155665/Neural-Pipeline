% Create custom cylinder with perforated base (with angled green holes)
figure('Position', [100, 100, 800, 600]);

% Parameters (in mm)
outer_radius = 9.5;       % 19mm diameter
inner_radius = 8.5;       % 1mm wall thickness
perforated_height = 3.5;
total_height = 14.0;
hole_spacing = 1.0;        % 1mm spacing between holes
hole_radius = 0.25;        % 0.5mm hole diameter

% Hole pattern: number of holes in each row (y-axis)
hole_rows = [5, 9, 11, 13, 15, 15, 17, 17, 17, 17, 17, 15, 13, 11, 9, 5];
num_rows = length(hole_rows);

% Angled hole parameters
angle_deg = 15;
angle_rad = deg2rad(angle_deg);
y_offset_per_height = tan(angle_rad) * perforated_height;
fprintf('Angled hole x-offset at top (z=%.1f mm): %.3f mm\n', perforated_height, y_offset_per_height); % check point

% Create a single 3D plot
hold on;
axis equal;

% Create outer cylinder (wall) with better visibility and detail
theta = linspace(0, 2*pi, 50); % Increased number of points for a smoother surface
z_outer = linspace(0, total_height, 40);
[Theta, Z] = meshgrid(theta, z_outer);

X_outer = outer_radius * cos(Theta);
Y_outer = outer_radius * sin(Theta);
surf(X_outer, Y_outer, Z, 'FaceColor', [0.5, 0.5, 0.5], 'FaceAlpha', 0.4, 'EdgeColor', 'none'); % Light blue cylinder with edges

% Create bottom plate (perforated section - 0 to 3.5mm)
z_plate = linspace(0, perforated_height, 15);
for z_val = z_plate
    r_circle = linspace(0, outer_radius, 20);
    theta_circle = linspace(0, 2*pi, 32);
    [R, Theta_circ] = meshgrid(r_circle, theta_circle);
    X_plate = R .* cos(Theta_circ);
    Y_plate = R .* sin(Theta_circ);
    Z_plate = ones(size(X_plate)) * z_val;
    surf(X_plate, Y_plate, Z_plate, 'FaceColor', [0.5, 0.5, 0.5], 'FaceAlpha', 0.05, 'EdgeColor', 'none'); % Gray color
end

% Create holes in the perforated section using the specified pattern
hole_count = 0;
center_row_idx = 8;  % Center the grid at (0, 0)

for row_idx = 1:num_rows
    num_holes_in_row = hole_rows(row_idx);
    % Y position: center hole row is at y=0
    y_pos = (row_idx - center_row_idx) * hole_spacing;
    
    % Calculate x positions for this row, centered at x=0
    x_center = (num_holes_in_row - 1) / 2.0 * hole_spacing;
    x_positions = linspace(-x_center, x_center, num_holes_in_row);
    
    for x_pos = x_positions
        distance_from_center = sqrt(x_pos^2 + y_pos^2);
        
        % Only create holes within outer radius
        if distance_from_center < outer_radius - 0.5  % Leave margin from edge
            % Determine hole color and type
            if (x_pos > 0 && y_pos > 0) || (x_pos + y_pos > 0)
                hole_color = 'g';  % green
                is_angled = true;
            else
                hole_color = 'r';  % red
                is_angled = false;
            end
            
            % Check if hole hits the wall
            if is_angled
                % Bottom of hole (z=0): original position
                dist_bottom = sqrt(x_pos^2 + y_pos^2) + hole_radius;
                % Top of hole (z=3.5mm): offset position
                dist_top = sqrt((y_pos + y_offset_per_height)^2 + x_pos^2) + hole_radius;

                % Skip this hole if the edge is within 2mm of the wall at any point
                if dist_bottom >= outer_radius - 2 || dist_top >= outer_radius - 2
                    continue;
                end
            end
            
            % Draw hole as a vertical cylinder (or angled for green holes)
            z_hole = linspace(0, perforated_height, 15);
            theta_hole = linspace(0, 2*pi, 16);
            [Theta_h, Z_h] = meshgrid(theta_hole, z_hole);
            
            if is_angled
                % For green holes, apply angular offset based on height
                z_ratio = Z_h / perforated_height;
                y_offset = y_offset_per_height * z_ratio;
                Y_h = (y_pos + y_offset) + hole_radius * cos(Theta_h);
            else
                Y_h = y_pos + hole_radius * cos(Theta_h);
            end
            
            X_h = x_pos + hole_radius * sin(Theta_h);
            surf(X_h, Y_h, Z_h, 'FaceColor', hole_color, 'FaceAlpha', 0.7, 'EdgeColor', hole_color, 'LineWidth', 0.5); % Add edges to holes
            hole_count = hole_count + 1;
        end
    end
end



xlabel('X (mm)', 'FontSize', 10);
ylabel('Y (mm)', 'FontSize', 10);
zlabel('Z (mm)', 'FontSize', 10);
title('Grid with Angled Green Holes (15°)', ...
      'FontSize', 11, 'FontWeight', 'bold');
xlim([-10, 10]);
ylim([-10, 10]);
zlim([0, total_height]);
view(45, 25);
grid on;

% Save the figure as a .fig file
savefig('grid_design_3d.fig');
