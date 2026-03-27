"""
fly_doublets_v3.py
==================
Fixed version - uses ATTITUDE_TARGET correctly and waits for
real altitude confirmation before proceeding.

Usage:
  Terminal 1 - start SITL and wait for "EKF3 IMU0 is using GPS":
    cd ~/Desktop/GSOC/ardupilot
    ./Tools/autotest/sim_vehicle.py -v ArduCopter --console --map \
        --out=udp:127.0.0.1:14550

  Terminal 2:
    python fly_doublets_v3.py
"""

import time
import math
import sys
from pymavlink import mavutil

CONNECT_STRING = 'udp:127.0.0.1:14550'
TAKEOFF_ALT    = 10.0
THRUST_HOVER   = 0.5


def connect():
    print("Connecting to SITL...")
    master = mavutil.mavlink_connection(CONNECT_STRING)
    master.wait_heartbeat(timeout=30)
    print(f"  Heartbeat OK - sysid={master.target_system}")
    return master


def set_mode(master, mode_name):
    mode_id = master.mode_mapping().get(mode_name)
    if mode_id is None:
        print(f"  ERROR: mode '{mode_name}' not found.")
        print(f"  Available modes: {list(master.mode_mapping().keys())}")
        sys.exit(1)
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    # Drain messages and wait for mode confirmation
    timeout = time.time() + 8
    while time.time() < timeout:
        msg = master.recv_match(blocking=True, timeout=1)
        if msg is None:
            continue
        if msg.get_type() == 'HEARTBEAT' and msg.custom_mode == mode_id:
            print(f"  Mode confirmed: {mode_name}")
            return
    print(f"  WARNING: mode {mode_name} ACK timeout - continuing anyway")


def arm(master):
    print("Arming...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,   # 1 = arm
        21196,  # magic number to force arm even if pre-arm checks fail
        0, 0, 0, 0, 0
    )
    timeout = time.time() + 10
    while time.time() < timeout:
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("  Armed OK")
            return
    print("  WARNING: arm timeout - trying again")
    # Second attempt without force
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    time.sleep(3)


def takeoff_and_wait(master, alt_m):
    print(f"Taking off to {alt_m}m...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt_m
    )
    # Wait properly - poll until stable at altitude for 2 consecutive seconds
    print("  Waiting for altitude...", flush=True)
    at_alt_since = None
    timeout = time.time() + 60
    while time.time() < timeout:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if msg is None:
            continue
        a = msg.relative_alt / 1000.0
        print(f"  alt: {a:.1f}m", end='\r', flush=True)
        if a >= alt_m * 0.90:
            if at_alt_since is None:
                at_alt_since = time.time()
            elif time.time() - at_alt_since >= 2.0:
                print(f"\n  Stable at {a:.1f}m - proceeding")
                return
        else:
            at_alt_since = None
    print(f"\n  WARNING: takeoff timeout at altitude {a:.1f}m - proceeding anyway")


def q_from_euler(roll_rad, pitch_rad, yaw_rad=0.0):
    cy = math.cos(yaw_rad   * 0.5); sy = math.sin(yaw_rad   * 0.5)
    cp = math.cos(pitch_rad * 0.5); sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad  * 0.5); sr = math.sin(roll_rad  * 0.5)
    return [
        cr*cp*cy + sr*sp*sy,   # w
        sr*cp*cy - cr*sp*sy,   # x
        cr*sp*cy + sr*cp*sy,   # y
        cr*cp*sy - sr*sp*cy,   # z
    ]


def send_att(master, roll_deg=0.0, pitch_deg=0.0, yaw_rate_dps=0.0, thrust=THRUST_HOVER):
    """
    SET_ATTITUDE_TARGET
    type_mask = 0b00000111 = use quaternion orientation + thrust,
                             ignore body rates (except yaw rate)
    """
    q = q_from_euler(math.radians(roll_deg), math.radians(pitch_deg))
    master.mav.set_attitude_target_send(
        int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms
        master.target_system,
        master.target_component,
        0b00000111,                  # type_mask: ignore roll/pitch rate, use yaw rate
        q,                           # quaternion [w, x, y, z]
        0.0,                         # roll rate  (ignored)
        0.0,                         # pitch rate (ignored)
        math.radians(yaw_rate_dps),  # yaw rate
        thrust                       # collective thrust [0..1]
    )


def hold(master, roll_deg=0.0, pitch_deg=0.0, yaw_rate_dps=0.0,
         duration=2.0, hz=25):
    """Send constant attitude at hz for duration seconds."""
    dt = 1.0 / hz
    n  = max(1, int(duration * hz))
    for _ in range(n):
        send_att(master, roll_deg=roll_deg, pitch_deg=pitch_deg,
                 yaw_rate_dps=yaw_rate_dps, thrust=THRUST_HOVER)
        time.sleep(dt)


def doublet(master, axis, angle_deg=15.0, hold_s=1.0, neutral_s=1.0):
    """
    One doublet: +angle for hold_s → neutral for neutral_s
                 -angle for hold_s → neutral for neutral_s
    """
    r = angle_deg if axis == 'roll'  else 0.0
    p = angle_deg if axis == 'pitch' else 0.0

    print(f"      +{angle_deg:.0f} deg  ({hold_s}s)")
    hold(master, roll_deg=r, pitch_deg=p, duration=hold_s)
    print(f"      neutral ({neutral_s}s)")
    hold(master, duration=neutral_s)
    print(f"      -{angle_deg:.0f} deg  ({hold_s}s)")
    hold(master, roll_deg=-r, pitch_deg=-p, duration=hold_s)
    print(f"      neutral ({neutral_s}s)")
    hold(master, duration=neutral_s)


def yaw_doublet(master, rate_dps=40.0, hold_s=1.2, neutral_s=1.0):
    print(f"      +{rate_dps:.0f} dps ({hold_s}s)")
    hold(master, yaw_rate_dps=rate_dps,   duration=hold_s)
    print(f"      neutral ({neutral_s}s)")
    hold(master, yaw_rate_dps=0,          duration=neutral_s)
    print(f"      -{rate_dps:.0f} dps ({hold_s}s)")
    hold(master, yaw_rate_dps=-rate_dps,  duration=hold_s)
    print(f"      neutral ({neutral_s}s)")
    hold(master, yaw_rate_dps=0,          duration=neutral_s)


def land(master):
    print("\nLanding...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    print("  Waiting 12s for touchdown...")
    time.sleep(12)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

master = connect()

print("\n--- STEP 1: Set GUIDED mode")
set_mode(master, 'GUIDED')
time.sleep(1)

print("\n--- STEP 2: Arm")
arm(master)
time.sleep(2)

print("\n--- STEP 3: Takeoff")
takeoff_and_wait(master, TAKEOFF_ALT)
time.sleep(1)

# ── BASELINE ─────────────────────────────────────────────────────────────────
print("\n--- STEP 4: Baseline hover (5s)")
hold(master, duration=5.0)

# ── ROLL DOUBLETS ─────────────────────────────────────────────────────────────
print("\n--- STEP 5: ROLL doublets 5x +/-20 deg  [excites Ixx]")
for i in range(5):
    print(f"  rep {i+1}/5")
    doublet(master, 'roll', angle_deg=20.0, hold_s=1.0, neutral_s=0.8)
hold(master, duration=3.0)

# ── PITCH DOUBLETS ────────────────────────────────────────────────────────────
print("\n--- STEP 6: PITCH doublets 5x +/-20 deg  [excites Iyy]")
for i in range(5):
    print(f"  rep {i+1}/5")
    doublet(master, 'pitch', angle_deg=20.0, hold_s=1.0, neutral_s=0.8)
hold(master, duration=3.0)

# ── YAW DOUBLETS ──────────────────────────────────────────────────────────────
print("\n--- STEP 7: YAW doublets 5x +/-40 dps  [excites Izz]")
for i in range(5):
    print(f"  rep {i+1}/5")
    yaw_doublet(master, rate_dps=40.0, hold_s=1.2, neutral_s=0.8)
hold(master, duration=3.0)

# ── LAND ──────────────────────────────────────────────────────────────────────
land(master)

print("\n[DONE]")
print("Find new log:  ls -lt ~/Desktop/GSOC/ardupilot/logs/*.BIN | head -3")
print("Verify it:     python scripts/verify_bin.py <newest>.BIN")