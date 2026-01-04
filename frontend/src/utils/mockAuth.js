// This is a mock implementation for development purposes
// Replace this file with actual authentication when connecting to a backend server

// Store mock users - these match the backend's default users
const mockUsers = [
  { username: 'officer', password: 'password', role: 'officer' },
  { username: 'admin', password: 'admin_password', role: 'admin' }
];

// Function to mock login
export const mockLogin = async (username, password) => {
  return new Promise((resolve, reject) => {
    // Simulate network delay
    setTimeout(() => {
      // Find user
      const user = mockUsers.find(
        user => user.username === username && user.password === password
      );
      
      if (user) {
        // Simulate JWT token
        const token = btoa(JSON.stringify({
          sub: user.username,
          role: user.role,
          exp: Date.now() + 3600000 // 1 hour expiry
        }));
        
        resolve({
          access_token: token,
          token_type: "bearer",
          user: {
            username: user.username,
            role: user.role
          }
        });
      } else {
        reject({
          response: {
            status: 401,
            data: {
              detail: "Incorrect username or password"
            }
          },
          message: "Authentication failed"
        });
      }
    }, 500); // 500ms delay to simulate network
  });
};