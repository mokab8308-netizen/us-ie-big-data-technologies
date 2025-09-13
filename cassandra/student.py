# student.py

import uuid
from datetime import date
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

# --- Health Check Conditions ---
MIN_AVG_WIND_SPEED = 15.0
MAX_AVG_WIND_SPEED = 25.0
MAX_ZERO_POWER_EVENTS = 5

def get_avg_wind_speed(session, turbine_id, reading_date):
    """Queries for the average wind speed for a given turbine and day."""
    query = SimpleStatement("""
        SELECT AVG(wind_speed)
        FROM turbine_daily_readings
        WHERE turbine_id = %s AND reading_date = %s;
    """)
    result = session.execute(query, (turbine_id, reading_date))
    row = result.one()
    # The result of AVG() is in the first column, index 0
    return row[0] if row else 0.0

def get_zero_power_events(session, turbine_id, reading_date):
    """Queries for the count of zero power events for a given turbine and day."""
    query = SimpleStatement("""
        SELECT COUNT(*)
        FROM zero_power_events
        WHERE turbine_id = %s AND reading_date = %s;
    """)
    result = session.execute(query, (turbine_id, reading_date))
    row = result.one()
    # The result of COUNT() is in the first column, index 0
    return row[0] if row else 0

def check_turbine_health(session, turbine_id, reading_date):
    """Determines if a turbine is healthy based on operational conditions."""
    print(f"--- Checking health for Turbine {turbine_id} on {reading_date} ---")

    # Get the required metrics by running the parametrized queries
    avg_speed = get_avg_wind_speed(session, turbine_id, reading_date)
    zero_events = get_zero_power_events(session, turbine_id, reading_date)

    print(f"Average Wind Speed: {avg_speed:.2f} m/s")
    print(f"Zero Power Events: {zero_events}")

    # Apply the health check logic
    is_speed_healthy = MIN_AVG_WIND_SPEED <= avg_speed <= MAX_AVG_WIND_SPEED
    is_event_count_healthy = zero_events <= MAX_ZERO_POWER_EVENTS

    # Determine and print the final status
    if is_speed_healthy and is_event_count_healthy:
        print("\n✅ Result: Turbine is Healthy")
    else:
        print("\n❌ Result: Turbine is Unhealthy")
        if not is_speed_healthy:
            print(f"   - Reason: Average wind speed ({avg_speed:.2f}) is outside the healthy range ({MIN_AVG_WIND_SPEED}-{MAX_AVG_WIND_SPEED} m/s).")
        if not is_event_count_healthy:
            print(f"   - Reason: Zero power event count ({zero_events}) exceeds the maximum allowed ({MAX_ZERO_POWER_EVENTS}).")

def main():
    """Main function to connect to Cassandra and run the health check."""
    try:
        cluster = Cluster(['127.0.0.1'], port=9042)
        session = cluster.connect('operations')
        print("Successfully connected to Cassandra.")
    except Exception as e:
        print(f"Failed to connect to Cassandra: {e}")
        return

    # --- PARAMETERS ---
    # Define the specific turbine and date to check.
    turbine_to_check = uuid.UUID('7f1a3d4e-5b6c-7d8a-9b1c-0a1b2c3d4e5f')
    date_to_check = date(2025, 9, 13)

    # Run the health check
    check_turbine_health(session, turbine_to_check, date_to_check)

    # Clean up the connection
    cluster.shutdown()
    print("\nConnection closed.")

if __name__ == "__main__":
    main()
