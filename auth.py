import streamlit as st
import hashlib
import hmac

def initialize_auth_state():
    """Initialize authentication related session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0

def get_users():
    """Get the dictionary of registered users
    
    In a real application, this would connect to a secure database
    """
    # Default admin credentials
    if 'users' not in st.session_state:
        st.session_state.users = {
            "admin": {
                "password": hash_password("admin123"),
                "role": "admin",
                "name": "Administrator"
            },
            "doctor": {
                "password": hash_password("doctor123"),
                "role": "doctor",
                "name": "Dr. Smith"
            },
            "nurse": {
                "password": hash_password("nurse123"),
                "role": "nurse",
                "name": "Nurse Johnson"
            }
        }
    
    return st.session_state.users

def hash_password(password):
    """Create a hashed password
    
    Args:
        password (str): Plain text password
        
    Returns:
        str: Hashed password
    """
    # In production, use a proper password hashing library with salt
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password, hashed_password):
    """Verify if a password matches the stored hash
    
    Args:
        plain_password (str): Password attempt
        hashed_password (str): Stored password hash
        
    Returns:
        bool: True if password matches
    """
    return hmac.compare_digest(
        hash_password(plain_password),
        hashed_password
    )

def login_user(username, password):
    """Attempt to log in a user
    
    Args:
        username (str): Username
        password (str): Password
        
    Returns:
        bool: True if login successful
    """
    users = get_users()
    
    # Check if username exists
    if username in users:
        # Verify password
        if verify_password(password, users[username]["password"]):
            st.session_state.logged_in = True
            st.session_state.user_role = users[username]["role"]
            st.session_state.user_name = users[username]["name"]
            st.session_state.current_username = username
            st.session_state.login_attempts = 0
            return True
    
    # Increment login attempts on failure
    st.session_state.login_attempts += 1
    return False

def logout_user():
    """Log out the current user"""
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.user_name = None
    st.session_state.current_username = None

def register_user(username, password, role="user", name=None):
    """Register a new user
    
    Args:
        username (str): Username
        password (str): Password
        role (str): User role
        name (str): Full name
        
    Returns:
        bool: True if registration successful
    """
    users = get_users()
    
    # Check if username already exists
    if username in users:
        return False
    
    # Add new user
    users[username] = {
        "password": hash_password(password),
        "role": role,
        "name": name if name else username
    }
    
    return True

def show_login_page():
    """Display the login interface
    
    Returns:
        bool: True if logged in successfully
    """
    st.title("Health Monitor AI - Login")
    
    # Create a clean login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if login_user(username, password):
                return True
            else:
                st.error(f"Invalid username or password. Attempts: {st.session_state.login_attempts}")
                return False
    
    # Display demo credentials
    with st.expander("Demo Credentials"):
        st.info("""
        For demonstration purposes, you can use:
        - Admin: username 'admin', password 'admin123'
        - Doctor: username 'doctor', password 'doctor123'
        - Nurse: username 'nurse', password 'nurse123'
        """)
    
    # Registration option
    st.markdown("---")
    if st.button("Register New Account"):
        st.session_state.show_registration = True
    
    # Show registration form if requested
    if st.session_state.get('show_registration', False):
        show_registration_form()
    
    return False

def show_registration_form():
    """Display the registration interface"""
    st.subheader("Register New Account")
    
    with st.form("registration_form"):
        new_username = st.text_input("Username", key="reg_username")
        new_password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password")
        full_name = st.text_input("Full Name")
        role = st.selectbox("Role", ["user", "nurse", "doctor"])
        
        register_button = st.form_submit_button("Register")
        
        if register_button:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif not new_username or not new_password:
                st.error("Username and password are required.")
            else:
                if register_user(new_username, new_password, role, full_name):
                    st.success("Registration successful! You can now log in.")
                    st.session_state.show_registration = False
                else:
                    st.error("Username already exists. Please choose a different username.")