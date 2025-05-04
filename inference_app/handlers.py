from sanic import Request, json


async def handle_list_events(request: Request):
    
    records = await request.app.ctx.database.get_records()
    
    return json({'estimated_count': len(records), 'records': records})
