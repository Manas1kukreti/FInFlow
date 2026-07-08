import React from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./auth/AuthContext.jsx";
import ProtectedRoute from "./auth/ProtectedRoute.jsx";
import AppShell from "./shell/AppShell.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import UploadCenter from "./pages/UploadCenter.jsx";
import SubmissionsPage from "./pages/SubmissionsPage.jsx";
import ManagerDashboard from "./pages/ManagerDashboard.jsx";
import AdminDashboard from "./pages/AdminDashboard.jsx";
import AuditPage from "./pages/AuditPage.jsx";
import SettingsPage from "./pages/SettingsPage.jsx";
import AlertsPage from "./pages/AlertsPage.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import LandingPage from "./pages/LandingPage.jsx";
import VerifyPasswordPage from "./pages/VerifyPasswordPage.jsx";
import AppErrorBoundary from "./components/AppErrorBoundary.jsx";
import "./styles.css";
import "./styles/design-system.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
    },
  },
});

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/login" element={<AuthPage />} />
              <Route path="/verify-password" element={<VerifyPasswordPage />} />

              <Route
                element={
                  <ProtectedRoute roles={["employee", "manager", "admin"]} />
                }
              >
                <Route element={<AppShell />}>
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/jobs" element={<SubmissionsPage />} />
                  <Route path="/jobs/new" element={<UploadCenter />} />
                  <Route path="/jobs/:jobId" element={<AuditPage />} />
                  <Route path="/settings" element={<SettingsPage />} />
                  <Route
                    path="/uploads"
                    element={<Navigate to="/jobs/new" replace />}
                  />
                  <Route
                    path="/submissions"
                    element={<Navigate to="/jobs" replace />}
                  />
                </Route>
              </Route>

              <Route element={<ProtectedRoute roles={["manager", "admin"]} />}>
                <Route element={<AppShell />}>
                  <Route path="/manager" element={<ManagerDashboard />} />
                  <Route path="/agents" element={<AdminDashboard />} />
                  <Route path="/alerts" element={<AlertsPage />} />
                </Route>
              </Route>

              <Route path="/admin" element={<Navigate to="/agents" replace />} />
              <Route path="/audit" element={<Navigate to="/jobs" replace />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </BrowserRouter>
        </AuthProvider>
      </QueryClientProvider>
    </AppErrorBoundary>
  </React.StrictMode>,
);
