package dto

type Response struct {
	Message string `json:"message"`
}

type JwtResponse struct {
	Token string `json:"token"`
}
