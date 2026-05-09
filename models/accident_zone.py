"""
RapidAid — Accident Zone Computation

Computes a tight, consistent bounding zone around the accident area.
Zone is based only on confirmed involved vehicles and victims.

Improvements:
  - Collision-point centering: zone centers on the actual impact point
    (intersection of involved vehicle bboxes), not just the merged bbox
  - Largest-vehicle bias: zone always includes the largest involved vehicle
  - Separate computation for vehicle-vehicle vs vehicle-pedestrian crashes
"""
from config import settings
from utils.geometry import (
    merge_boxes, expand_box, compute_diagonal, compute_box_area,
    compute_box_center
)


class AccidentZoneCalculator:
    """Computes the accident zone from involved vehicles and victims."""

    def compute(self, involved_vehicles, victims, frame_width, frame_height,
                collision_point=None):
        """
        Compute accident zone bounding box.

        Args:
            involved_vehicles: list of involved vehicle dicts
            victims: list of victim dicts
            frame_width: frame width in pixels
            frame_height: frame height in pixels
            collision_point: optional (x, y) tuple for the estimated
                impact point. If provided, zone is centered on this point.

        Returns:
            list [x1, y1, x2, y2] or None if no entities
        """
        all_boxes = [v["bbox"] for v in involved_vehicles]
        all_boxes += [p["bbox"] for p in victims]

        if not all_boxes:
            return None

        # If we have a collision point, center the zone on it
        if collision_point is not None:
            return self._compute_centered_zone(
                collision_point, all_boxes, frame_width, frame_height
            )

        # Try to estimate collision point from vehicle intersections
        est_collision = self._estimate_collision_point(involved_vehicles, victims)
        if est_collision is not None:
            return self._compute_centered_zone(
                est_collision, all_boxes, frame_width, frame_height
            )

        # Fallback: standard merged bbox approach
        return self._compute_standard_zone(all_boxes, frame_width, frame_height)

    def _estimate_collision_point(self, involved_vehicles, victims):
        """
        Estimate where the actual collision occurred.

        For vehicle-vehicle crashes: intersection center of overlapping bboxes.
        For vehicle-pedestrian crashes: center of victim bbox.
        For single vehicle: center of the vehicle bbox.

        Returns:
            tuple (x, y) or None
        """
        # Case 1: Vehicle-pedestrian crash (victims present)
        if victims and involved_vehicles:
            # Collision point is between the victim and the nearest involved vehicle
            victim_centers = [compute_box_center(v["bbox"]) for v in victims]
            vehicle_centers = [compute_box_center(v["bbox"]) for v in involved_vehicles]

            # Average of victim centers and nearest vehicle center
            avg_vx = sum(c[0] for c in victim_centers) / len(victim_centers)
            avg_vy = sum(c[1] for c in victim_centers) / len(victim_centers)

            # Find nearest vehicle to victim cluster
            best_vc = vehicle_centers[0]
            best_dist = float('inf')
            for vc in vehicle_centers:
                dist = ((vc[0] - avg_vx)**2 + (vc[1] - avg_vy)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_vc = vc

            # Collision point is midway between victim cluster and nearest vehicle
            cx = (avg_vx + best_vc[0]) / 2
            cy = (avg_vy + best_vc[1]) / 2
            return (cx, cy)

        # Case 2: Vehicle-vehicle crash (2+ vehicles)
        if len(involved_vehicles) >= 2:
            # Try to find bbox intersection of the two largest vehicles
            sorted_vehicles = sorted(
                involved_vehicles,
                key=lambda v: compute_box_area(v["bbox"]),
                reverse=True
            )

            v1 = sorted_vehicles[0]["bbox"]
            v2 = sorted_vehicles[1]["bbox"]

            # Compute intersection rectangle
            ix1 = max(v1[0], v2[0])
            iy1 = max(v1[1], v2[1])
            ix2 = min(v1[2], v2[2])
            iy2 = min(v1[3], v2[3])

            if ix1 < ix2 and iy1 < iy2:
                # Bboxes overlap — collision point is center of intersection
                return ((ix1 + ix2) / 2, (iy1 + iy2) / 2)
            else:
                # Bboxes don't overlap — collision point is midpoint between
                # the closest edges
                c1 = compute_box_center(v1)
                c2 = compute_box_center(v2)
                return ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)

        # Case 3: Single vehicle
        if len(involved_vehicles) == 1:
            return compute_box_center(involved_vehicles[0]["bbox"])

        # Case 4: Victims only (hit and run)
        if victims:
            centers = [compute_box_center(v["bbox"]) for v in victims]
            avg_x = sum(c[0] for c in centers) / len(centers)
            avg_y = sum(c[1] for c in centers) / len(centers)
            return (avg_x, avg_y)

        return None

    def _compute_centered_zone(self, collision_point, all_boxes,
                                frame_width, frame_height):
        """
        Compute zone centered on the collision point, sized to encompass
        all involved entities.
        """
        cx, cy = collision_point

        # Compute the max distance from collision point to any entity edge
        max_dist_x = 0
        max_dist_y = 0
        for box in all_boxes:
            dx1 = abs(box[0] - cx)
            dx2 = abs(box[2] - cx)
            dy1 = abs(box[1] - cy)
            dy2 = abs(box[3] - cy)
            max_dist_x = max(max_dist_x, dx1, dx2)
            max_dist_y = max(max_dist_y, dy1, dy2)

        # Add padding
        pad_x = max(frame_width * settings.ZONE_PADDING_RATIO,
                     max_dist_x * 0.15)
        pad_y = max(frame_height * settings.ZONE_PADDING_RATIO,
                     max_dist_y * 0.15)

        # Build zone centered on collision point
        half_w = max_dist_x + pad_x
        half_h = max_dist_y + pad_y

        zone = [
            int(max(0, cx - half_w)),
            int(max(0, cy - half_h)),
            int(min(frame_width, cx + half_w)),
            int(min(frame_height, cy + half_h)),
        ]

        # Enforce minimum zone size
        zone = self._enforce_min_size(zone, frame_width, frame_height)

        # Enforce maximum zone size
        zone = self._enforce_max_size(zone, frame_width, frame_height)

        return zone

    def _compute_standard_zone(self, all_boxes, frame_width, frame_height):
        """Standard merged-bbox zone computation (fallback)."""
        merged = merge_boxes(all_boxes)
        if merged is None:
            return None

        # Compute adaptive padding
        zone_w = merged[2] - merged[0]
        zone_h = merged[3] - merged[1]

        pad_x = int(max(frame_width * settings.ZONE_PADDING_RATIO,
                        zone_w * 0.1))
        pad_y = int(max(frame_height * settings.ZONE_PADDING_RATIO,
                        zone_h * 0.1))

        # Expand with padding
        zone = expand_box(merged, pad_x, pad_y, frame_width, frame_height)

        zone = self._enforce_min_size(zone, frame_width, frame_height)
        zone = self._enforce_max_size(zone, frame_width, frame_height)

        return zone

    def _enforce_min_size(self, zone, frame_width, frame_height):
        """Enforce minimum zone size."""
        min_w = int(frame_width * settings.MIN_ZONE_SIZE_RATIO)
        min_h = int(frame_height * settings.MIN_ZONE_SIZE_RATIO)
        actual_w = zone[2] - zone[0]
        actual_h = zone[3] - zone[1]

        if actual_w < min_w:
            extra = (min_w - actual_w) // 2
            zone = expand_box(zone, extra, 0, frame_width, frame_height)

        if actual_h < min_h:
            extra = (min_h - actual_h) // 2
            zone = expand_box(zone, 0, extra, frame_width, frame_height)

        return zone

    def _enforce_max_size(self, zone, frame_width, frame_height):
        """Enforce maximum zone size."""
        max_w = int(frame_width * settings.MAX_ZONE_SIZE_RATIO)
        max_h = int(frame_height * settings.MAX_ZONE_SIZE_RATIO)

        if (zone[2] - zone[0]) > max_w or (zone[3] - zone[1]) > max_h:
            cx = (zone[0] + zone[2]) // 2
            cy = (zone[1] + zone[3]) // 2
            half_w = min(max_w, zone[2] - zone[0]) // 2
            half_h = min(max_h, zone[3] - zone[1]) // 2
            zone = [
                max(0, cx - half_w),
                max(0, cy - half_h),
                min(frame_width, cx + half_w),
                min(frame_height, cy + half_h),
            ]

        return zone
